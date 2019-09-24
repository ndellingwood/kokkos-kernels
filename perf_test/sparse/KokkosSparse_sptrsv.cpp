/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include <cusparse.h>
#endif

#include <Kokkos_Core.hpp>
#include <matrix_market.hpp>

#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_sptrsv.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_IOUtils.hpp>

// Testing compilation with lapacke for dtrtri usage
//#define KOKKOSKERNELS_SPTRSV_LAPACKE_TRSV
//#define CHECKALLRUNRESULTS

#ifdef KOKKOSKERNELS_SPTRSV_LAPACKE_TRSV
#include "lapacke.h"
#endif

//#define PRINTVIEWSSPTRSVPERF

#if defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA ) && (!defined(KOKKOS_ENABLE_CUDA) || ( 8000 <= CUDA_VERSION ))
using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

//#define PRINT_DENSETRIMTX
//#define PRINT_HLEVEL_FREQ_PLOT
//#define PRINT_LEVEL_LIST

enum {DEFAULT, CUSPARSE, LVLSCHED_RP, LVLSCHED_TP1, LVLSCHED_TP2, LVLSCHED_TP1CHAIN, LVLSCHED_DENSEP_TP1, LVLSCHED_DENSEP_TP2};

#ifdef PRINTVIEWSSPTRSVPERF
template <class ViewType>
void print_view1d(const ViewType dv) {
  auto v = Kokkos::create_mirror_view(dv);
  Kokkos::deep_copy(v,dv);
  std::cout << "Output for view " << v.label() << std::endl;
  for (size_t i = 0; i < v.extent(0); ++i) {
    std::cout << "v(" << i << ") = " << v(i) << " , ";
  }
  std::cout << std::endl;
}
#endif

template<typename Scalar>
int test_sptrsv_perf(std::vector<int> tests, const std::string& lfilename, const std::string& ufilename, const int team_size, const int vector_length, const int idx_offset, const int loop, const int chain_threshold = 0, const float dense_row_percent = -1.0) {

  typedef Scalar scalar_t;
  typedef int lno_t;
  typedef int size_type;
  typedef Kokkos::DefaultExecutionSpace execution_space;
  typedef typename execution_space::memory_space memory_space;

  typedef KokkosSparse::CrsMatrix<scalar_t, lno_t, execution_space, void, size_type> crsmat_t;
  typedef typename crsmat_t::StaticCrsGraphType graph_t;

  typedef Kokkos::View< scalar_t*, memory_space >     ValuesType;

  typedef KokkosKernels::Experimental::KokkosKernelsHandle <size_type, lno_t, scalar_t,
    execution_space, memory_space, memory_space > KernelHandle;

  const scalar_t ZERO = scalar_t(0);
  const scalar_t ONE  = scalar_t(1);


// Read lmtx
// Run all requested algorithms
// Read umtx
// Run all requested algorithms

// LOWERTRI
  std::cout << "\n\n" << std::endl;
  if (!lfilename.empty())
  {
    std::cout << "Lower Tri Begin: Read matrix filename " << lfilename << std::endl;
    crsmat_t triMtx = KokkosKernels::Impl::read_kokkos_crst_matrix<crsmat_t>(lfilename.c_str()); //in_matrix
    graph_t  graph  = triMtx.graph; // in_graph
    const size_type nrows = graph.numRows();

    // Create the rhs and lhs_known solution
    ValuesType known_lhs("known_lhs", nrows);
    // Create known solution lhs set to all 1's
    Kokkos::deep_copy(known_lhs, ONE);

    // Solution to find
    ValuesType lhs("lhs", nrows);

    // A*known_lhs generates rhs: rhs is dense, use spmv
    ValuesType rhs("rhs", nrows);

    std::cout << "SPMV" << std::endl;
    KokkosSparse::spmv( "N", ONE, triMtx, known_lhs, ZERO, rhs);


    auto row_map = graph.row_map;
    auto entries = graph.entries;
    auto values  = triMtx.values;

    std::cout << "Lower Perf: row_map.extent(0) = " << row_map.extent(0) << std::endl;
    std::cout << "Lower Perf: entries.extent(0) = " << entries.extent(0) << std::endl;
    std::cout << "Lower Perf: values.extent(0) = " << values.extent(0) << std::endl;

    std::cout << "Lower Perf: lhs.extent(0) = " << lhs.extent(0) << std::endl;
    std::cout << "Lower Perf: rhs.extent(0) = " << rhs.extent(0) << std::endl;

#ifdef PRINTVIEWSSPTRSVPERF
    print_view1d(row_map);
    print_view1d(entries);
    print_view1d(values);
    print_view1d(known_lhs);
    print_view1d(rhs);
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
  //std::cout << "  cusparse: create handle" << std::endl;
  cusparseStatus_t status;
  cusparseHandle_t handle = 0;
  status = cusparseCreate(&handle);
  if (CUSPARSE_STATUS_SUCCESS != status)
    std::cout << "handle create status error name " << (status) << std::endl;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cusparseMatDescr_t descr = 0;
  csrsv2Info_t info = 0;
  int pBufferSize;
  void *pBuffer = 0;
  int structural_zero;
  int numerical_zero;
  const double alpha = 1.;
  const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  
  // step 1: create a descriptor which contains
  // - matrix L is lower triangular
  //   (L may not have all diagonal elements.)
  status = cusparseCreateMatDescr(&descr);
  if (CUSPARSE_STATUS_SUCCESS != status)
    std::cout << "matdescr create status error name " << (status) << std::endl;
  //cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  //cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);
  
  // step 2: create a empty info structure
  //std::cout << "  cusparse: create csrsv2info" << std::endl;
  status = cusparseCreateCsrsv2Info(&info);
  if (CUSPARSE_STATUS_SUCCESS != status)
    std::cout << "csrsv2info create status error name " << (status) << std::endl;
  
  // step 3: query how much memory used in csrsv2, and allocate the buffer
        int nnz = triMtx.nnz();
  cusparseDcsrsv2_bufferSize(handle, trans, nrows, nnz, descr,
      values.data(), row_map.data(), entries.data(), info, &pBufferSize);
  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  cudaMalloc((void**)&pBuffer, pBufferSize);
#endif


  for ( auto test : tests ) {
    std::cout << "\ntest = " << test << std::endl;

    KernelHandle kh;
    bool is_lower_tri = true;

    std::cout << "Create handle (lower)" << std::endl;
    switch(test) {
      case LVLSCHED_RP:
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_RP, nrows, is_lower_tri);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1:
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_TP1, nrows, is_lower_tri);
        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1CHAIN:
        printf("TP1 with CHAIN\n");
        printf("chain_threshold %d\n", chain_threshold);
        printf("team_size %d\n", team_size);
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_TP1CHAIN, nrows, is_lower_tri);
        kh.get_sptrsv_handle()->reset_chain_threshold(chain_threshold);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP2:
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHED_TP2, nrows, is_lower_tri);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        if (vector_length != -1) kh.get_sptrsv_handle()->set_vector_size(vector_length);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_DENSEP_TP1:
        printf("dense_row_percent %f\n", dense_row_percent);
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1, nrows, is_lower_tri);
        kh.get_sptrsv_handle()->reset_chain_threshold(chain_threshold);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        if (dense_row_percent != -1) kh.get_sptrsv_handle()->set_dense_partition_row_percent(dense_row_percent);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_DENSEP_TP2:
        printf("dense_row_percent %f\n", dense_row_percent);
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP2, nrows, is_lower_tri);
        kh.get_sptrsv_handle()->reset_chain_threshold(chain_threshold);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        if (vector_length != -1) kh.get_sptrsv_handle()->set_vector_size(vector_length);
        if (dense_row_percent != -1) kh.get_sptrsv_handle()->set_dense_partition_row_percent(dense_row_percent);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case CUSPARSE:
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
        std::cout << "CUSPARSE: No kk interface added yet" << std::endl;
        //cusparse_matvec(A, x, y, rows_per_thread, team_size, vector_length);
        break;
#else
        std::cout << "CUSPARSE not enabled: Fall through to defaults" << std::endl;
#endif
      default:
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_TP1, nrows, is_lower_tri);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        kh.get_sptrsv_handle()->print_algorithm();
    }

#ifdef KOKKOSKERNELS_SPTRSV_LAPACKE_TRSV
    if (test==LVLSCHED_DENSEP_TP1 || test==LVLSCHED_DENSEP_TP2) {
      // Extract dense triangle from matrix, copy to host for dtrtri

      auto thandle = kh.get_sptrsv_handle();
      auto dense_partition_nrows = thandle->get_dense_partition_nrows();

      typedef Kokkos::View<scalar_t**, memory_space>     DenseTriType;
      DenseTriType dense_trimtx("dense_trimtx", dense_partition_nrows, dense_partition_nrows);

      auto dense_row_start = thandle->get_dense_partition_row_start();
      auto trimtx_col_start = is_lower_tri ? dense_row_start : 0; // ends at nrows

      auto dprow_map = Kokkos::subview( row_map, Kokkos::pair<size_type,size_type>(dense_row_start, dense_row_start+dense_partition_nrows+1) );

    Kokkos::parallel_for("fill dense tri", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dense_partition_nrows),
    KOKKOS_LAMBDA (const size_type i) {
      // Iterate over partition of original matrix to extract the indices in the rectangular sparse matrix partition
      size_type offset_start = dprow_map(i);
      size_type offset_end   = dprow_map(i+1);

      //auto idx_count_this_row = row_map_rectspmtx(i+1) - row_map_rectspmtx(i);
      //auto new_idx_offset = row_map_rectspmtx(i);

      for (size_type offset = offset_start; offset < offset_end; ++offset) {
        size_type colid = entries(offset);
        auto val = values(offset);
        // Count in-sparse-rect entries per row, store at row_map(rowid+1) in anticipation of followup scan
        //if ( (is_lower_tri && colid < trimtx_col_start) || (!is_lower_tri && colid >= rectspmtx_col_start) )
        if ( (is_lower_tri && colid < trimtx_col_start) )
        {
          //values_rectspmtx(new_idx_offset) = val;
          //++new_idx_offset;
        }
        else {
          auto trimtx_shifted_colid = is_lower_tri ? colid - trimtx_col_start : colid;
          dense_trimtx(i, trimtx_shifted_colid) = val;
        }
      }

    });
    Kokkos::fence();

    auto hdense_trimtx = Kokkos::create_mirror_view(dense_trimtx);
    Kokkos::deep_copy(hdense_trimtx, dense_trimtx);

    #ifdef PRINT_DENSETRIMTX
    // Print result to output file, check with Matlab
    // Print the actual matrix
     {
      std::ofstream outfile;
      outfile.open("L.txt");
      if (outfile.is_open()) {
        for ( size_t i = 0; i < hdense_trimtx.extent(0); ++i ) {
          for ( size_t j = 0; j < hdense_trimtx.extent(1); ++j ) {
            //outfile << i << " " << j << " " << hdense_trimtx(i,j) << std::endl;
            if (j < hdense_trimtx.extent(1)-1) {
              outfile << hdense_trimtx(i,j) << " ";
            }
            else {
              outfile << hdense_trimtx(i,j) << std::endl;
            }
          }
          //outfile << std::endl;
        }
        outfile.close();
      }
      else {
        std::cout << "L OUTFILE DID NOT OPEN!!!" << std::endl;
      }
     }
    #endif

    // If hdense_trimtx is LayoutLeft
//    LAPACKE_dtrtri(LAPACK_COL_MAJOR,
//                  'L', 'N', (int)dense_partition_nrows, (scalar_t*)hdense_trimtx.data(), (int)hdense_trimtx.stride_0()); // Stride is final entry

    // If hdense_trimtx is LayoutRight
     LAPACKE_dtrtri(LAPACK_ROW_MAJOR,
                   'L', 'N', (int)dense_partition_nrows, (scalar_t*)hdense_trimtx.data(), (int)hdense_trimtx.stride_0()); // Stride is final entry


    #ifdef PRINT_DENSETRIMTX
    // Print result to output file, check with Matlab
    // Print the actual inverse matrix
     {
      std::ofstream outfile;
      outfile.open("Linv.txt");
      if (outfile.is_open()) {
        for ( size_t i = 0; i < hdense_trimtx.extent(0); ++i ) {
          for ( size_t j = 0; j < hdense_trimtx.extent(1); ++j ) {
            //outfile << i << " " << j << " " << hdense_trimtx(i,j) << std::endl;
            if (j < hdense_trimtx.extent(1)-1) {
              outfile << hdense_trimtx(i,j) << " ";
            }
            else {
              outfile << hdense_trimtx(i,j) << std::endl;
            }
          }
          //outfile << std::endl;
        }
        outfile.close();
      }
      else {
        std::cout << "Linv OUTFILE DID NOT OPEN!!!" << std::endl;
      }
     }
    #endif

    }
#endif


    // Init run to clear cache etc.
    Kokkos::Timer timer;
    if (test != CUSPARSE) {
    timer.reset();
    sptrsv_symbolic( &kh, row_map, entries );
    std::cout << "LTRI Symbolic Time: " << timer.seconds() << std::endl;

    //std::cout << "TriSolve Solve" << std::endl;
    timer.reset();
    sptrsv_solve( &kh, row_map, entries, values, rhs, lhs );
    Kokkos::fence();
    std::cout << "LTRI Solve Time: " << timer.seconds() << std::endl;
  
    }
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
// step 4: perform analysis
    else {
      //int nnz = triMtx.nnz();
      //std::cout << "  cusparse path: analysis" << std::endl;
      //status = cusparseDcsrsv2_analysis(handle, trans, nrows, nnz, descr, (double*)dvalues, (int *)drow_map, (int *)dentries, info, policy, pBuffer);
      timer.reset();
      status = cusparseDcsrsv2_analysis(handle, trans, nrows, triMtx.nnz(), descr, values.data(), row_map.data(), entries.data(), info, policy, pBuffer);
      std::cout << "LTRI Cusparse Symbolic Time: " << timer.seconds() << std::endl;
      if (CUSPARSE_STATUS_SUCCESS != status)
        std::cout << "analysis status error name " << (status) << std::endl;
// L has unit diagonal, so no structural zero is reported.

      //std::cout << "  cusparse path: analysis" << std::endl;
      status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
      if (CUSPARSE_STATUS_ZERO_PIVOT == status){
         printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
      }

// step 5: solve L*y = x
      //std::cout << "  cusparse path: solve" << std::endl;
      //status = cusparseDcsrsv2_solve(handle, trans, nrows, nnz, &alpha, descr, (double*)dvalues, (int *)drow_map, (int *)dentries, info, (double*)drhs, (double*)dlhs, policy, pBuffer);
      timer.reset();
      status = cusparseDcsrsv2_solve(handle, trans, nrows, triMtx.nnz(), &alpha, descr, values.data(), row_map.data(), entries.data(), info, rhs.data(), lhs.data(), policy, pBuffer);
      Kokkos::fence();
      std::cout << "LTRI Cusparse Solve Time: " << timer.seconds() << std::endl;
      if (CUSPARSE_STATUS_SUCCESS != status)
        std::cout << "solve status error name " << (status) << std::endl;
// L has unit diagonal, so no numerical zero is reported.
      status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
      if (CUSPARSE_STATUS_ZERO_PIVOT == status){
         printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
      }
    }
#endif
    // Error Check
    Kokkos::fence();
    {
    scalar_t sum = 0.0;
    Kokkos::parallel_reduce( Kokkos::RangePolicy<execution_space>(0, lhs.extent(0)), 
      KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
        tsum += lhs(i);
      }, sum);
  
    if ( sum != lhs.extent(0) ) {
      std::cout << "Lower Tri Solve FAILURE: sum = " << sum << std::endl;
      auto hsoln = Kokkos::create_mirror_view(lhs);
      Kokkos::deep_copy(hsoln, lhs);
      for ( size_t i = 0; i < hsoln.extent(0); ++i ) {
        std::cout << "lhs(" << i << ") = " << hsoln(i) << std::endl;
      }
      return 1;
    }
    else {
     std::cout << "\nLower Tri Solve Init Test: SUCCESS!\n" << std::endl;
    }
    }

  
    // Benchmark
    Kokkos::fence();
    double min_time = 1.0e32;
    double max_time = 0.0;
    double ave_time = 0.0;

    for(int i=0;i<loop;i++) {
      timer.reset();
  
    if (test != CUSPARSE) {
      sptrsv_solve( &kh, row_map, entries, values, rhs, lhs );
    #ifdef CHECKALLRUNRESULTS
        {
        scalar_t sum = 0.0;
        Kokkos::parallel_reduce( Kokkos::RangePolicy<execution_space>(0, lhs.extent(0)), 
          KOKKOS_LAMBDA ( const lno_t it, scalar_t &tsum ) {
            tsum += lhs(it);
          }, sum);
      
        if ( sum != lhs.extent(0) ) {
          std::cout << "Lower Tri Solve FAILURE: sum = " << sum << std::endl;
          auto hsoln = Kokkos::create_mirror_view(lhs);
          Kokkos::deep_copy(hsoln, lhs);
          for ( size_t it = 0; it < hsoln.extent(0); ++it ) {
            std::cout << "lhs(" << it << ") = " << hsoln(it) << std::endl;
          }
          return 1;
        }
        else {
         std::cout << "\nLower Tri Solve Init Test: SUCCESS!\n" << std::endl;
        }
        }
    #endif
    }
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
    else {
      cusparseDcsrsv2_solve(handle, trans, nrows, triMtx.nnz(), &alpha, descr, values.data(), row_map.data(), entries.data(), info, rhs.data(), lhs.data(), policy, pBuffer);
    }
#endif
  
      Kokkos::fence();
      double time = timer.seconds();
      ave_time += time;
      if(time>max_time) max_time = time;
      if(time<min_time) min_time = time;
    }

    std::cout << "LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
    std::cout << "LOOP_MAX_TIME:  " << max_time << std::endl;
    std::cout << "LOOP_MIN_TIME:  " << min_time << std::endl;

    // Output for level frequency plot
    #ifdef PRINT_HLEVEL_FREQ_PLOT
    {
    auto hnpl = kh.get_sptrsv_handle()->get_host_nodes_per_level();
    auto nlevels = kh.get_sptrsv_handle()->get_num_levels();
    std::string algmstring = kh.get_sptrsv_handle()->return_algorithm_string();
    std::cout << algmstring << std::endl;
    // Create filename
    std::string filename = "lower_nodes_per_level_" + algmstring + ".txt";
    std::cout << filename << std::endl;
    std::cout << "  nlevels = " << nlevels << std::endl;
    std::ofstream outfile;
    outfile.open(filename);
    if (outfile.is_open()) {
      for ( int i = 0; i < nlevels; ++i ) {
        outfile << hnpl(i) << std::endl;
        //std::cout  << hnpl(i) << std::endl;
      }
      outfile.close();
    }
    else {
      std::cout << "OUTFILE DID NOT OPEN!!!" << std::endl;
    }

    auto hngpl = kh.get_sptrsv_handle()->get_host_nodes_grouped_by_level();
    filename = "lower_nodes_groupby_level_" + algmstring + ".txt";
    std::cout << filename << std::endl;
    outfile.open(filename);
    if (outfile.is_open()) {
      for ( size_t i = 0; i < hngpl.extent(0); ++i )
        outfile << hngpl(i) << std::endl;
      outfile.close();
    }
    else {
      std::cout << "OUTFILE DID NOT OPEN!!!" << std::endl;
    }

    auto htree = kh.get_sptrsv_handle()->get_host_dep_tree();
    filename = "lower_htree_" + algmstring + ".txt";
    std::cout << filename << std::endl;
    outfile.open(filename);
    if (outfile.is_open()) {
      for ( size_t i = 0; i < htree.extent(0); ++i )
        outfile << htree(i) << std::endl;
      outfile.close();
    }
    else {
      std::cout << "OUTFILE DID NOT OPEN!!!" << std::endl;
    }

    }
    #endif

    #ifdef PRINT_LEVEL_LIST
    {
    auto level_list = kh.get_sptrsv_handle()->get_level_list();
    auto hlevel_list = Kokkos::create_mirror_view(level_list);
    Kokkos::deep_copy(hlevel_list, level_list);

    auto nlevels = kh.get_sptrsv_handle()->get_num_levels();

    std::string algmstring = kh.get_sptrsv_handle()->return_algorithm_string();
    std::cout << algmstring << std::endl;
    // Create filename
    std::string filename = "lower_level_list_" + algmstring + ".txt";
    std::cout << filename << std::endl;
    std::cout << "  nlevels = " << nlevels << "  nodes = " << hlevel_list.extent(0) << std::endl;
    std::ofstream outfile;
    outfile.open(filename);
    if (outfile.is_open()) {
      for ( size_t i = 0; i < hlevel_list.extent(0); ++i )
        outfile << hlevel_list(i) << std::endl;
      outfile.close();
    }
    else {
      std::cout << "OUTFILE DID NOT OPEN!!!" << std::endl;
    }
    }
    #endif

    kh.destroy_sptrsv_handle();
  }

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
// step 6: free resources
    cudaFree(pBuffer);
    cusparseDestroyCsrsv2Info(info);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
#endif
  } // end lowertri

  Kokkos::fence();
  std::cout << "\n\n" << std::endl;
// UPPERTRI
  if (!ufilename.empty())
  {
    std::cout << "Upper Tri Begin: Read matrix filename " << ufilename << std::endl;
    crsmat_t triMtx = KokkosKernels::Impl::read_kokkos_crst_matrix<crsmat_t>(ufilename.c_str()); //in_matrix
    graph_t  graph  = triMtx.graph; // in_graph
    const size_type nrows = graph.numRows();

    // Create the rhs and lhs_known solution
    ValuesType known_lhs("known_lhs", nrows);
    // Create known solution lhs set to all 1's
    Kokkos::deep_copy(known_lhs, ONE);

    // Solution to find
    ValuesType lhs("lhs", nrows);

    // A*known_lhs generates rhs: rhs is dense, use spmv
    ValuesType rhs("rhs", nrows);

    std::cout << "SPMV" << std::endl;
    KokkosSparse::spmv( "N", ONE, triMtx, known_lhs, ZERO, rhs);

    auto row_map = graph.row_map;
    auto entries = graph.entries;
    auto values  = triMtx.values;

    std::cout << "Upper Perf: row_map.extent(0) = " << row_map.extent(0) << std::endl;
    std::cout << "Upper Perf: entries.extent(0) = " << entries.extent(0) << std::endl;
    std::cout << "Upper Perf: values.extent(0) = " << values.extent(0) << std::endl;

    std::cout << "Upper Perf: lhs.extent(0) = " << lhs.extent(0) << std::endl;
    std::cout << "Upper Perf: rhs.extent(0) = " << rhs.extent(0) << std::endl;

#ifdef PRINTVIEWSSPTRSVPERF
    print_view1d(row_map);
    print_view1d(entries);
    print_view1d(values);
    print_view1d(known_lhs);
    print_view1d(rhs);
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
  //std::cout << "  cusparse: create handle" << std::endl;
  cusparseStatus_t status;
  cusparseHandle_t handle = 0;
  status = cusparseCreate(&handle);
  if (CUSPARSE_STATUS_SUCCESS != status)
    std::cout << "handle create status error name " << (status) << std::endl;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cusparseMatDescr_t descr = 0;
  csrsv2Info_t info = 0;
  int pBufferSize;
  void *pBuffer = 0;
  int structural_zero;
  int numerical_zero;
  const double alpha = 1.;
  const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  
  // step 1: create a descriptor which contains
  //   (L may not have all diagonal elements.)
  status = cusparseCreateMatDescr(&descr);
  if (CUSPARSE_STATUS_SUCCESS != status)
    std::cout << "matdescr create status error name " << (status) << std::endl;
  //cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  //cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  //cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);
  
  // step 2: create a empty info structure
  //std::cout << "  cusparse: create csrsv2info" << std::endl;
  status = cusparseCreateCsrsv2Info(&info);
  if (CUSPARSE_STATUS_SUCCESS != status)
    std::cout << "csrsv2info create status error name " << (status) << std::endl;
  
  // step 3: query how much memory used in csrsv2, and allocate the buffer
        int nnz = triMtx.nnz();
  cusparseDcsrsv2_bufferSize(handle, trans, nrows, nnz, descr,
      values.data(), row_map.data(), entries.data(), info, &pBufferSize);
  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  cudaMalloc((void**)&pBuffer, pBufferSize);
#endif


  for ( auto test : tests ) {
    std::cout << "\ntest = " << test << std::endl;

    KernelHandle kh;
    bool is_lower_tri = false;

    std::cout << "Create handle (upper)" << std::endl;
    switch(test) {
      case LVLSCHED_RP:
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_RP, nrows, is_lower_tri);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1:
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_TP1, nrows, is_lower_tri);
        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1CHAIN:
        printf("TP1 with CHAIN\n");
        printf("chain_threshold %d\n", chain_threshold);
        printf("team_size %d\n", team_size);
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_TP1CHAIN, nrows, is_lower_tri);
        kh.get_sptrsv_handle()->reset_chain_threshold(chain_threshold);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP2:
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHED_TP2, nrows, is_lower_tri);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        if (vector_length != -1) kh.get_sptrsv_handle()->set_vector_size(vector_length);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_DENSEP_TP1:
        printf("dense_row_percent %f\n", dense_row_percent);
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1, nrows, is_lower_tri);
        kh.get_sptrsv_handle()->reset_chain_threshold(chain_threshold);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        if (dense_row_percent != -1) kh.get_sptrsv_handle()->set_dense_partition_row_percent(dense_row_percent);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_DENSEP_TP2:
        printf("dense_row_percent %f\n", dense_row_percent);
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP2, nrows, is_lower_tri);
        kh.get_sptrsv_handle()->reset_chain_threshold(chain_threshold);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        if (vector_length != -1) kh.get_sptrsv_handle()->set_vector_size(vector_length);
        if (dense_row_percent != -1) kh.get_sptrsv_handle()->set_dense_partition_row_percent(dense_row_percent);
        kh.get_sptrsv_handle()->print_algorithm();
        break;
      case CUSPARSE:
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
        std::cout << "CUSPARSE: No kk interface added yet" << std::endl;
        //cusparse_matvec(A, x, y, rows_per_thread, team_size, vector_length);
        break;
#else
        std::cout << "CUSPARSE not enabled: Fall through to defaults" << std::endl;
#endif
      default:
        kh.create_sptrsv_handle(SPTRSVAlgorithm::SEQLVLSCHD_TP1, nrows, is_lower_tri);
        if (team_size != -1) kh.get_sptrsv_handle()->set_team_size(team_size);
        kh.get_sptrsv_handle()->print_algorithm();
    }


    // Init run to clear cache etc.
    Kokkos::Timer timer;
    if (test != CUSPARSE) {
    timer.reset();
    sptrsv_symbolic( &kh, row_map, entries );
    std::cout << "UTRI Symbolic Time: " << timer.seconds() << std::endl;

    //std::cout << "TriSolve Solve" << std::endl;
    timer.reset();
    sptrsv_solve( &kh, row_map, entries, values, rhs, lhs );
    Kokkos::fence();
    std::cout << "UTRI Solve Time: " << timer.seconds() << std::endl;
  
    }
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
// step 4: perform analysis
    else {
      //int nnz = triMtx.nnz();
      //std::cout << "  cusparse path: analysis" << std::endl;
      //status = cusparseDcsrsv2_analysis(handle, trans, nrows, nnz, descr, (double*)dvalues, (int *)drow_map, (int *)dentries, info, policy, pBuffer);
      timer.reset();
      status = cusparseDcsrsv2_analysis(handle, trans, nrows, triMtx.nnz(), descr, values.data(), row_map.data(), entries.data(), info, policy, pBuffer);
      std::cout << "UTRI Cusparse Symbolic Time: " << timer.seconds() << std::endl;
      if (CUSPARSE_STATUS_SUCCESS != status)
        std::cout << "analysis status error name " << (status) << std::endl;
      // L has unit diagonal, so no structural zero is reported.

      status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
      if (CUSPARSE_STATUS_ZERO_PIVOT == status){
         printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
      }

// step 5: solve L*y = x
      //std::cout << "  cusparse path: solve" << std::endl;
      //status = cusparseDcsrsv2_solve(handle, trans, nrows, nnz, &alpha, descr, (double*)dvalues, (int *)drow_map, (int *)dentries, info, (double*)drhs, (double*)dlhs, policy, pBuffer);
      timer.reset();
      status = cusparseDcsrsv2_solve(handle, trans, nrows, triMtx.nnz(), &alpha, descr, values.data(), row_map.data(), entries.data(), info, rhs.data(), lhs.data(), policy, pBuffer);
      Kokkos::fence();
      std::cout << "UTRI Cusparse Solve Time: " << timer.seconds() << std::endl;
      if (CUSPARSE_STATUS_SUCCESS != status)
        std::cout << "solve status error name " << (status) << std::endl;
      // L has unit diagonal, so no numerical zero is reported.
      status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
      if (CUSPARSE_STATUS_ZERO_PIVOT == status){
         printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
      }
    }
#endif
    // Error Check
    Kokkos::fence();
    {
    scalar_t sum = 0.0;
    Kokkos::parallel_reduce( Kokkos::RangePolicy<execution_space>(0, lhs.extent(0)), 
      KOKKOS_LAMBDA ( const lno_t i, scalar_t &tsum ) {
        tsum += lhs(i);
      }, sum);
  
    if ( sum != scalar_t(lhs.extent(0)) ) {
      std::cout << "Upper Tri Solve FAILURE: sum = " << sum << std::endl;
      auto hsoln = Kokkos::create_mirror_view(lhs);
      Kokkos::deep_copy(hsoln, lhs);
      for ( size_t i = 0; i < hsoln.extent(0); ++i ) {
        std::cout << "lhs(" << i << ") = " << hsoln(i) << std::endl;
      }
      return 1;
    }
    else {
     std::cout << "\nUpper Tri Solve Init Test: SUCCESS!\n" << std::endl;
    }
    }
  
    // Benchmark
    Kokkos::fence();
    double min_time = 1.0e32;
    double max_time = 0.0;
    double ave_time = 0.0;

    for(int i=0;i<loop;i++) {
      timer.reset();
  
    if (test != CUSPARSE) {
      sptrsv_solve( &kh, row_map, entries, values, rhs, lhs );
    #ifdef CHECKALLRUNRESULTS
        {
        scalar_t sum = 0.0;
        Kokkos::parallel_reduce( Kokkos::RangePolicy<execution_space>(0, lhs.extent(0)), 
          KOKKOS_LAMBDA ( const lno_t it, scalar_t &tsum ) {
            tsum += lhs(it);
          }, sum);
      
        if ( sum != scalar_t(lhs.extent(0)) ) {
          std::cout << "Upper Tri Solve FAILURE: sum = " << sum << std::endl;
          auto hsoln = Kokkos::create_mirror_view(lhs);
          Kokkos::deep_copy(hsoln, lhs);
          for ( size_t it = 0; it < hsoln.extent(0); ++it ) {
            std::cout << "lhs(" << it << ") = " << hsoln(it) << std::endl;
          }
          return 1;
        }
        else {
         std::cout << "\nUpper Tri Solve Init Test: SUCCESS!\n" << std::endl;
        }
        }
    #endif
    }
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
    else {
      cusparseDcsrsv2_solve(handle, trans, nrows, triMtx.nnz(), &alpha, descr, values.data(), row_map.data(), entries.data(), info, rhs.data(), lhs.data(), policy, pBuffer);
    }
#endif
  
      Kokkos::fence();
      double time = timer.seconds();
      ave_time += time;
      if(time>max_time) max_time = time;
      if(time<min_time) min_time = time;
    }

    std::cout << "LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
    std::cout << "LOOP_MAX_TIME:  " << max_time << std::endl;
    std::cout << "LOOP_MIN_TIME:  " << min_time << std::endl;

    // Output for level frequency plot
    #ifdef PRINT_HLEVEL_FREQ_PLOT
    {
    auto hnpl = kh.get_sptrsv_handle()->get_host_nodes_per_level();
    auto nlevels = kh.get_sptrsv_handle()->get_num_levels();
    std::string algmstring = kh.get_sptrsv_handle()->return_algorithm_string();
    std::cout << algmstring << std::endl;
    // Create filename
    std::string filename = "upper_nodes_per_level_" + algmstring + ".txt";
    std::cout << filename << std::endl;
    std::cout << "  nlevels = " << nlevels << std::endl;
    std::ofstream outfile;
    outfile.open(filename);
    if (outfile.is_open()) {
      for ( int i = 0; i < nlevels; ++i ) {
        outfile << hnpl(i) << std::endl;
        //std::cout  << hnpl(i) << std::endl;
      }
      outfile.close();
    }
    else {
      std::cout << "OUTFILE DID NOT OPEN!!!" << std::endl;
    }

    auto hngpl = kh.get_sptrsv_handle()->get_host_nodes_grouped_by_level();
    filename = "lower_nodes_groupby_level_" + algmstring + ".txt";
    std::cout << filename << std::endl;
    outfile.open(filename);
    if (outfile.is_open()) {
      for ( size_t i = 0; i < hngpl.extent(0); ++i )
        outfile << hngpl(i) << std::endl;
      outfile.close();
    }
    else {
      std::cout << "OUTFILE DID NOT OPEN!!!" << std::endl;
    }

    auto htree = kh.get_sptrsv_handle()->get_host_dep_tree();
    filename = "upper_htree_" + algmstring + ".txt";
    std::cout << filename << std::endl;
    outfile.open(filename);
    if (outfile.is_open()) {
      for ( size_t i = 0; i < htree.extent(0); ++i )
        outfile << htree(i) << std::endl;
      outfile.close();
    }
    else {
      std::cout << "OUTFILE DID NOT OPEN!!!" << std::endl;
    }
    }
    #endif
    #ifdef PRINT_LEVEL_LIST
    {
    auto level_list = kh.get_sptrsv_handle()->get_level_list();
    auto hlevel_list = Kokkos::create_mirror_view(level_list);
    Kokkos::deep_copy(hlevel_list, level_list);

    auto nlevels = kh.get_sptrsv_handle()->get_num_levels();

    std::string algmstring = kh.get_sptrsv_handle()->return_algorithm_string();
    std::cout << algmstring << std::endl;
    // Create filename
    std::string filename = "lower_level_list_" + algmstring + ".txt";
    std::cout << filename << std::endl;
    std::cout << "  nlevels = " << nlevels << "  nodes = " << hlevel_list.extent(0) << std::endl;
    std::ofstream outfile;
    outfile.open(filename);
    if (outfile.is_open()) {
      for ( size_t i = 0; i < hlevel_list.extent(0); ++i )
        outfile << hlevel_list(i) << std::endl;
      outfile.close();
    }
    else {
      std::cout << "OUTFILE DID NOT OPEN!!!" << std::endl;
    }
    }
    #endif

    kh.destroy_sptrsv_handle();
  }

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
// step 6: free resources
    cudaFree(pBuffer);
    cusparseDestroyCsrsv2Info(info);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
#endif
  } // end uppertri
  Kokkos::fence();

  return 0;
}


void print_help_sptrsv() {
  printf("Options:\n");
  printf("  --test [OPTION] : Use different kernel implementations\n");
  printf("                    Options:\n");
  printf("                      lvlrp, lvltp1, lvltp2, lvltp1chain, lvldensetp1, lvldensetp2\n\n");
  printf("                      cusparse           (Vendor Libraries)\n\n");
  printf("  -lf [file]      : Read in Matrix Market formatted text file 'file'.\n");
  printf("  -uf [file]      : Read in Matrix Market formatted text file 'file'.\n");
  printf("  --offset [O]    : Subtract O from every index.\n");
  printf("                    Useful in case the matrix market file is not 0 based.\n\n");
  printf("  -ts [T]         : Number of threads per team.\n");
  printf("  -vl [V]         : Vector-length (i.e. how many Cuda threads are a Kokkos 'thread').\n");
  printf("  -ct [V]         : Chain threshold: Only has effect of lvltp1chain algorithm.\n");
  printf("  -dr [V]         : Dense row percent (as float): Only has effect of lvldensetp1 algorithm.\n");
  printf("  --loop [LOOP]   : How many spmv to run to aggregate average time. \n");
//  printf("  --write-lvl-freq: Write output files with number of nodes per level for each matrix and algorithm.\n");
//  printf("  -s [N]          : generate a semi-random banded (band size 0.01xN) NxN matrix\n");
//  printf("                    with average of 10 entries per row.\n");
//  printf("  --schedule [SCH]: Set schedule for kk variant (static,dynamic,auto [ default ]).\n");
//  printf("  -fb [file]      : Read in binary Matrix files 'file'.\n");
//  printf("  --write-binary  : In combination with -f, generate binary files.\n");
}


int main(int argc, char **argv)
{
 std::vector<int> tests;

 std::string lfilename;
 std::string ufilename;

 int vector_length = -1;
 int team_size = -1;
 int idx_offset = 0;
 int loop = 1;
 int chain_threshold = 0;
 float dense_row_percent = -1.0;
// int schedule=AUTO;

 if(argc == 1) {
   print_help_sptrsv();
   return 0;
 }

 for(int i=0;i<argc;i++)
 {
  if((strcmp(argv[i],"--test")==0)) {
    i++;
    if((strcmp(argv[i],"lvlrp")==0)) {
      tests.push_back( LVLSCHED_RP );
    }
    if((strcmp(argv[i],"lvltp1")==0)) {
      tests.push_back( LVLSCHED_TP1 );
    }
    if((strcmp(argv[i],"lvltp1chain")==0)) {
      tests.push_back( LVLSCHED_TP1CHAIN );
    }
    if((strcmp(argv[i],"lvltp2")==0)) {
      tests.push_back( LVLSCHED_TP2 );
    }
    if((strcmp(argv[i],"lvldensetp1")==0)) {
      tests.push_back( LVLSCHED_DENSEP_TP1 );
    }
    if((strcmp(argv[i],"lvldensetp2")==0)) {
      tests.push_back( LVLSCHED_DENSEP_TP2 );
    }
    if((strcmp(argv[i],"cusparse")==0)) {
      tests.push_back( CUSPARSE );
    }
    continue;
  }
  if((strcmp(argv[i],"-lf")==0)) {lfilename = argv[++i]; continue;}
  if((strcmp(argv[i],"-uf")==0)) {ufilename = argv[++i]; continue;}
  if((strcmp(argv[i],"-ts")==0)) {team_size=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-vl")==0)) {vector_length=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-ct")==0)) {chain_threshold=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-dr")==0)) {dense_row_percent=atof(argv[++i]); continue;}
  if((strcmp(argv[i],"-l")==0)) {loop=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--offset")==0)) {idx_offset=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--loop")==0)) {loop=atoi(argv[++i]); continue;}
/*
  if((strcmp(argv[i],"-lfb")==0)) {lfilename = argv[++i]; binaryfile = true; continue;}
  if((strcmp(argv[i],"-ufb")==0)) {ufilename = argv[++i]; binaryfile = true; continue;}
  if((strcmp(argv[i],"--schedule")==0)) {
    i++;
    if((strcmp(argv[i],"auto")==0))
      schedule = AUTO;
    if((strcmp(argv[i],"dynamic")==0))
      schedule = DYNAMIC;
    if((strcmp(argv[i],"static")==0))
      schedule = STATIC;
    continue;
  }
*/
  if((strcmp(argv[i],"--help")==0) || (strcmp(argv[i],"-h")==0)) {
    print_help_sptrsv();
    return 0;
  }
 }

 if (tests.size() == 0) {
   tests.push_back(DEFAULT);
 }
 for (size_t i = 0; i < tests.size(); ++i) {
    std::cout << "tests[" << i << "] = " << tests[i] << std::endl;
 }


 Kokkos::initialize(argc,argv);
 {
   int total_errors = test_sptrsv_perf<double>(tests, lfilename, ufilename, team_size, vector_length, idx_offset, loop, chain_threshold, dense_row_percent);

   if(total_errors == 0)
   printf("Kokkos::SPTRSV Test: Passed\n");
   else
   printf("Kokkos::SPTRSV Test: Failed\n");


  }
  Kokkos::finalize();
  return 0;
}
#else
int main() {
  std::cout << "KokkosSparse_sptrsv: This perf_test will do nothing when Cuda is enabled without lambda support." << std::endl;
  return 0;
}
#endif
