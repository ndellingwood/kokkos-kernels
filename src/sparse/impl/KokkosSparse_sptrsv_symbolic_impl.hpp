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

#ifndef KOKKOSSPARSE_IMPL_SPTRSV_SYMBOLIC_HPP_
#define KOKKOSSPARSE_IMPL_SPTRSV_SYMBOLIC_HPP_

/// \file Kokkos_Sparse_impl_sptrsv_symbolic.hpp
/// \brief Implementation(s) of sparse triangular solve.

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_sptrsv_handle.hpp>

#define TRISOLVE_SYMB_TIMERS
//#define LVL_OUTPUT_INFO
//#define CHAIN_LVL_OUTPUT_INFO
//#define PRINT1DVIEWSSYMB
//#define DEBUGSYMBDENSE
#define SYMB_DIAG_CHECK
//#define SYMB_INIT_ASSUME_LVL

// TODO Pass values array and store diagonal entries - should this always be done or optional?

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

#ifdef PRINT1DVIEWSSYMB
template <class ViewType>
void print_view1d_symbolic(const ViewType dv) {
  auto v = Kokkos::create_mirror_view(dv);
  Kokkos::deep_copy(v, dv);
  std::cout << "Output for view " << v.label() << std::endl;
  for (size_t i = 0; i < v.extent(0); ++i) {
    std::cout << "v(" << i << ") = " << v(i) << " , ";
  }
  std::cout << std::endl;
}
#endif

template < class TriSolveHandle, class NPLViewType >
void symbolic_chain_phase(TriSolveHandle &thandle, const NPLViewType &nodes_per_level) {

#ifdef TRISOLVE_SYMB_TIMERS
  Kokkos::Timer timer_sym_chain_total;
#endif
  typedef typename TriSolveHandle::size_type size_type;

  size_type level = thandle.get_num_levels();

  // Create the chain now
  // FIXME Implementations will need to be templated on exec space it seems...
 auto cutoff_threshold = thandle.get_chain_threshold();
 //thandle.print_algorithm();
 if ( thandle.algm_requires_symb_chain() ) {
  std::cout << "SYMB Call CHAIN version" << std::endl;
  auto h_chain_ptr = thandle.get_host_chain_ptr();
  h_chain_ptr(0) = 0;
  size_type chain_length = 0;
  size_type num_chain_entries = 0;
  int update_chain = 0;
  //const int cutoff = std::is_same<typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>::value ? 1 : 256; // TODO chain cutoff hard-coded to 256: make this a "threshold" parameter in the handle
  const int cutoff = cutoff_threshold;
  std::cout << "  cutoff_thresh = " << cutoff << std::endl;
  for ( size_type i = 0; i < level; ++i ) {
    auto cnpl = nodes_per_level(i);
    //std::cout << "incre chain_length  npl(" << i << ") = " << nodes_per_level(i) << std::endl;
    if (cnpl <= cutoff) {
      // this level may be part of a chain passed to the "single_block" solver to reduce kernel launches
      chain_length += 1;
    }
    else {
      // Too many levels to run on single block...
      // If first lvl <= cutoff but next level isn't, the two aren't separately updated and info is lost...
      // if chain_length > 0, take path so that chain-links updated, then current too large chain updated (i.e. 2 updates); if chain_length == 0, then no previous chains and only one update required (npl too large for single-block
      update_chain = chain_length > 0 ? 2 : 1;
    }

    // if we hit final level before a trigger to update the chain, than override it - in this case, there was not a larger value to miss cutoff and reset the update
    if ( update_chain == 0 && i == level-1 ) { update_chain = 1; }


    if (update_chain == 1) {
      num_chain_entries += 1;
      //std::cout << "  nce = " << num_chain_entries << "  chain_length = " << chain_length << std::endl;
      if (chain_length == 0) {
        h_chain_ptr(num_chain_entries) = h_chain_ptr(num_chain_entries-1) + 1;
      }
      else {
        h_chain_ptr(num_chain_entries) = h_chain_ptr(num_chain_entries-1) + chain_length;
      }
      chain_length = 0; //reset
      update_chain = 0; //reset
    }

    // Two updates required - should only occur if chain_length > 0
    if (update_chain == 2) {
      if (chain_length == 0) { std::cout << "MAJOR LOGIC ERROR! TERMINATE!" << std::endl; exit(-1); }

      num_chain_entries += 1;
      h_chain_ptr(num_chain_entries) = h_chain_ptr(num_chain_entries-1) + chain_length;
      //std::cout << "  nce = " << num_chain_entries << "  chain_length = " << chain_length << std::endl;

      num_chain_entries += 1;
      h_chain_ptr(num_chain_entries) = h_chain_ptr(num_chain_entries-1) + 1;

      chain_length = 0; //reset
      update_chain = 0; //reset
    }
  }
  thandle.set_num_chain_entries(num_chain_entries);
#ifdef CHAIN_LVL_OUTPUT_INFO
  std::cout << "  num_chain_entries = " << thandle.get_num_chain_entries() << std::endl;
  for ( size_type i = 0; i < num_chain_entries+1; ++i )
  {
    std::cout << "chain_ptr(" << i << "): " << h_chain_ptr(i) << std::endl;
  }
#endif
 }
  // Usage:
  // for c in [0, num_chain_entries)
  //   s = h_chain_ptr(c); e = h_chain_ptr(c+1);
  //   num_levels_in_current_chain = e - s;
  //   if nlicc > 256
  //     call current_alg
  //   else
  //     call single_block(s,e)

#ifdef TRISOLVE_SYMB_TIMERS
 std::cout << "  Symbolic Chain Phase Total Time: " << timer_sym_chain_total.seconds() << std::endl;;
#endif
} // end symbolic_chain_phase





template < class TriSolveHandle, class RowMapType, class EntriesType >
void lower_tri_symbolic (TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries) {
#ifdef TRISOLVE_SYMB_TIMERS
  Kokkos::Timer timer_sym_lowertri_total;
#endif
  std::cout << "  Begin lower_tri symbolic" << std::endl;

 if ( thandle.algm_requires_symb_lvlsched() )
 {
  // Scheduling currently compute on host - need host copy of all views

  typedef typename TriSolveHandle::size_type size_type;

  typedef typename TriSolveHandle::nnz_lno_view_t  DeviceEntriesType;
  //typedef typename TriSolveHandle::nnz_lno_view_t::HostMirror HostEntriesType;

  typedef typename TriSolveHandle::signed_nnz_lno_view_t DeviceSignedEntriesType;
  typedef typename TriSolveHandle::signed_nnz_lno_view_t::HostMirror HostSignedEntriesType;

  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

//  size_type nrows = thandle.get_nrows();
  // Necessary for partitioned persisting sparse matrix
  size_type nrows = drow_map.extent(0)-1;

  auto row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(row_map, drow_map);

  auto entries = Kokkos::create_mirror_view(dentries);
  Kokkos::deep_copy(entries, dentries);
  
  // get device view - will deep_copy to it at end of this host routine
  DeviceEntriesType dnodes_per_level = thandle.get_nodes_per_level();
// FIXME Use handles hnodes_per_level
  auto nodes_per_level = thandle.get_host_nodes_per_level();
//  HostEntriesType nodes_per_level = Kokkos::create_mirror_view(dnodes_per_level);
//  Kokkos::deep_copy(nodes_per_level, dnodes_per_level);

  // get device view - will deep_copy to it at end of this host routine
  DeviceEntriesType dnodes_grouped_by_level = thandle.get_nodes_grouped_by_level();
// FIXME Use handles hnodes_grouped_by_level
  auto nodes_grouped_by_level = thandle.get_host_nodes_grouped_by_level();
//  HostEntriesType nodes_grouped_by_level = Kokkos::create_mirror_view(dnodes_grouped_by_level);
//  Kokkos::deep_copy(nodes_grouped_by_level, dnodes_grouped_by_level);

  DeviceSignedEntriesType dlevel_list = thandle.get_level_list();
  HostSignedEntriesType level_list = Kokkos::create_mirror_view(dlevel_list);
  Kokkos::deep_copy(level_list, dlevel_list);

  HostSignedEntriesType previous_level_list( Kokkos::ViewAllocateWithoutInitializing("previous_level_list"), nrows );
  Kokkos::deep_copy( previous_level_list, signed_integral_t(-1) );

  auto diagonal_offsets = thandle.get_diagonal_offsets();
  // diagonal_offsets is uninitialized - deep_copy unnecessary at the beginning, only needed at the end
  auto hdiagonal_offsets = Kokkos::create_mirror_view(diagonal_offsets);

  size_type level = 0;

#ifdef SYMB_DIAG_CHECK
  long diag_ctr = 0;
#endif

#ifdef DENSEPARTITION
  //auto starting_node = thandle.get_lvlsched_node_start();
  //auto ending_node = thandle.get_lvlsched_node_end();
  auto starting_node = 0;
  auto ending_node = nrows;
#else
  auto starting_node = 0;
  auto ending_node = nrows;
#endif

  // FIXME Change this to allow for partitioned sparse mtx
#ifdef SYMB_INIT_ASSUME_LVL
  // node 0 is trivially independent in lower tri solve, start with it in level 0
  level_list(starting_node) = level;
  size_type node_count = 1; //lower tri: starting with node 0 already in level 0

  nodes_per_level(0) = 1;
  nodes_grouped_by_level(0) = starting_node;
  hdiagonal_offsets(starting_node) = starting_node; //FIXME Add to upper impl
#else
  size_type node_count = 0;
#endif

  while (node_count < nrows) {

#ifdef SYMB_INIT_ASSUME_LVL
    for ( size_type row = starting_node+1; row < ending_node; ++row ) { // row 0 already included
#else
    for ( size_type row = starting_node; row < ending_node; ++row ) {
#endif
      if ( level_list(row) == -1 ) { // unmarked
        bool is_root = true;
        signed_integral_t ptrstart = row_map(row);
        signed_integral_t ptrend   = row_map(row+1);

        for (signed_integral_t offset = ptrstart; offset < ptrend; ++offset) {
          size_type col = entries(offset); //FIXME: Can col be incorrectly mapped, wrong range compared to shifted re-start row_map?
          // FIXME: For lower_tri, colid is unchanged; shifted for upper_tri...
          if ( previous_level_list(col) == -1 && col != row ) { // unmarked
            if ( col < row ) {
              is_root = false;
              break;
            }
          }
          else if ( col == row ) {
            //FIXME Not reliable to ensure all entries stored yet
            //TODO if loop breaks before this is found, this may not get stored...
            //TODO possibly store/sort in same order as the nodes in the level_list
            //TODO Maybe run FULL check the first round through without breaking in upper if statement...
            hdiagonal_offsets(row) = offset;
#ifdef SYMB_DIAG_CHECK
            ++diag_ctr;
            // FIXME: The starting_node index is skipped, must be manually included
#endif
          }
          else if ( col > row ) {
            std::cout << "SYMB ERROR: Lower tri with colid > rowid - SHOULD NOT HAPPEN!!!";
            std::cout << "\nrow = " << row << "  col = " << col << "  offset = " << offset << std::endl;
            exit(-1);
          }
        } // end for offset , i.e. cols of this row

        if ( is_root == true ) {
          level_list(row) = level;
          nodes_per_level(level) += 1;
          nodes_grouped_by_level(node_count) = row;
          node_count += 1;
        }

      } // end if
    } // end for row

    //Kokkos::deep_copy(previous_level_list, level_list);
    for ( size_type i = 0; i < nrows; ++i ) {
      previous_level_list(i) = level_list(i);
    }

    level += 1;
    //std::cout << "  node_count = " << node_count << "  level = " << level << "  nrows = " << nrows << std::endl;
  } // end while

  thandle.set_num_levels(level);
#ifdef SYMB_DIAG_CHECK
    std::cout << "  SYMB: diag_ctr = " << diag_ctr << "  nrows = " << nrows << std::endl;
#endif

  // Create the chain now
  if ( thandle.algm_requires_symb_chain() ) {
    std::cout << "  Symbolic chain phase begin" << std::endl;
    symbolic_chain_phase(thandle, nodes_per_level);
  }

  thandle.set_symbolic_complete();

  // Output check
#ifdef LVL_OUTPUT_INFO
  std::cout << "  set symbolic complete: " << thandle.is_symbolic_complete() << std::endl;
  std::cout << "  set num levels: " << thandle.get_num_levels() << std::endl;

  std::cout << "  lower_tri_symbolic result: " << std::endl;
  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "node: " << i << "  level_list = " << level_list(i) << std::endl; }

  for ( size_type i = 0; i < level; ++i )
  { std::cout << "level: " << i << "  nodes_per_level = " << nodes_per_level(i) << std::endl; }

  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "i: " << i << "  nodes_grouped_by_level = " << nodes_grouped_by_level(i) << std::endl; }
#endif

  // Deep copy to device views
  Kokkos::deep_copy(dnodes_grouped_by_level, nodes_grouped_by_level);
  Kokkos::deep_copy(dnodes_per_level, nodes_per_level);
  Kokkos::deep_copy(dlevel_list, level_list);
  Kokkos::deep_copy(diagonal_offsets, hdiagonal_offsets);
 }

#ifdef TRISOLVE_SYMB_TIMERS
 std::cout << "  Symbolic (lower tri) Total Time: " << timer_sym_lowertri_total.seconds() << std::endl;;
#endif
} // end lower_tri_symbolic


template < class TriSolveHandle, class RowMapType, class EntriesType >
void upper_tri_symbolic ( TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries ) {
#ifdef TRISOLVE_SYMB_TIMERS
  Kokkos::Timer timer_sym_uppertri_total;
#endif

  std::cout << "  Begin upper_tri symbolic" << std::endl;

 if ( thandle.algm_requires_symb_lvlsched() )
 {
  // Scheduling currently compute on host - need host copy of all views

  typedef typename TriSolveHandle::size_type size_type;

  typedef typename TriSolveHandle::nnz_lno_view_t  DeviceEntriesType;
  //typedef typename TriSolveHandle::nnz_lno_view_t::HostMirror HostEntriesType;

  typedef typename TriSolveHandle::signed_nnz_lno_view_t DeviceSignedEntriesType;
  typedef typename TriSolveHandle::signed_nnz_lno_view_t::HostMirror HostSignedEntriesType;

  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

//  size_type nrows = thandle.get_nrows();
  // Necessary for partitioned persisting sparse matrix
  size_type nrows = drow_map.extent(0)-1;

  auto row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(row_map, drow_map);

  auto entries = Kokkos::create_mirror_view(dentries);
  Kokkos::deep_copy(entries, dentries);
  
  // get device view - will deep_copy to it at end of this host routine
  DeviceEntriesType dnodes_per_level = thandle.get_nodes_per_level();
// FIXME Use handles hnodes_per_level
  auto nodes_per_level = thandle.get_host_nodes_per_level();
//  HostEntriesType nodes_per_level = Kokkos::create_mirror_view(dnodes_per_level);
//  Kokkos::deep_copy(nodes_per_level, dnodes_per_level);

  // get device view - will deep_copy to it at end of this host routine
  DeviceEntriesType dnodes_grouped_by_level = thandle.get_nodes_grouped_by_level();
// FIXME Use handles hnodes_grouped_by_level
  auto nodes_grouped_by_level = thandle.get_host_nodes_grouped_by_level();
//  HostEntriesType nodes_grouped_by_level = Kokkos::create_mirror_view(dnodes_grouped_by_level);
//  Kokkos::deep_copy(nodes_grouped_by_level, dnodes_grouped_by_level);

  DeviceSignedEntriesType dlevel_list = thandle.get_level_list();
  HostSignedEntriesType level_list = Kokkos::create_mirror_view(dlevel_list);
  Kokkos::deep_copy(level_list, dlevel_list);

  HostSignedEntriesType previous_level_list( Kokkos::ViewAllocateWithoutInitializing("previous_level_list"), nrows);
  Kokkos::deep_copy( previous_level_list, signed_integral_t(-1) );

  auto diagonal_offsets = thandle.get_diagonal_offsets();
  // diagonal_offsets is uninitialized - deep_copy unnecessary at the beginning, only needed at the end
  auto hdiagonal_offsets = Kokkos::create_mirror_view(diagonal_offsets);

  size_type level = 0;
  // FIXME: starting_node only holds for FULL matrix, not the sparse partition in different algorithm
  // Depending on algorithm, can be nrows - 1 vs dense_start_row - 1
  // FIXME Change this to allow for partitioned sparse mtx
#ifdef DENSEPARTITION
  auto dense_nrows = thandle.get_dense_partition_nrows();
  auto starting_node = nrows - 1;
  auto ending_node = 0;
  //auto starting_node = thandle.get_lvlsched_node_start();
  //auto ending_node = thandle.get_lvlsched_node_end();
#else
  auto starting_node = nrows - 1;
  auto ending_node = 0;
#endif
  std::cout << "  upper_tri_symbolic debug: " << std::endl;
  std::cout << "  starting_node = " << starting_node << "  ending_node = " << ending_node << "  nrows = " << nrows << std::endl;

#ifdef SYMB_INIT_ASSUME_LVL
  // final row is trivially independent in upper tri solve, start with it in level 0
  level_list(starting_node) = level;
  size_type node_count = 1; //upper tri: starting with node n already in level 0

  nodes_per_level(0) = 1;
  nodes_grouped_by_level(0) = starting_node;
#else
  size_type node_count = 0;
#endif

  while (node_count < nrows) {

#ifdef SYMB_INIT_ASSUME_LVL
    for ( signed_integral_t row = starting_node-1; row >= ending_node; --row ) { // row 0 already included
#else
    for ( signed_integral_t row = starting_node; row >= ending_node; --row ) {
#endif
      if ( level_list(row) == -1 ) { // unmarked
        bool is_root = true;
        signed_integral_t ptrstart = row_map(row);
        signed_integral_t ptrend   = row_map(row+1);

        for (signed_integral_t offset = ptrend-1; offset >= ptrstart; --offset) {
#ifdef DENSEPARTITION
          signed_integral_t original_col = entries(offset);
          signed_integral_t col = original_col - dense_nrows;
          //signed_integral_t col = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri - don't need to check on type of tri mtx since called only in upper impl
#else
          signed_integral_t col = entries(offset);
#endif

          // FIXME: This check for col != row vs col > row will be broken with partial persist_sptrimtx, since using a row_map with rowid shifted to 0 but not colid
          // May need to fix row instead, using the shift, so that level_list stores the original rowid for the solve??
          // FIXME: Or, is it that the solve needs similar colid adjustment as done here????
          if ( previous_level_list(col) == -1 && col != row ) { // unmarked
            if ( col > row ) {
              is_root = false;
              break;
            }
          }
          else if ( col == row ) {
            //FIXME Not reliable to ensure all entries stored yet
            //TODO if loop breaks before this is found, this may not get stored...
            //TODO possibly store/sort in same order as the nodes in the level_list
            //TODO Maybe run FULL check the first round through without breaking in upper if statement...
            hdiagonal_offsets(row) = offset;
          }
        } // end for offset , i.e. cols of this row

        if ( is_root == true ) {
          level_list(row) = level;
          nodes_per_level(level) += 1;
          nodes_grouped_by_level(node_count) = row;
          node_count += 1;
        }

      } // end if
    } // end for row

    //Kokkos::deep_copy(previous_level_list, level_list);
    for ( size_type i = 0; i < nrows; ++i ) {
      previous_level_list(i) = level_list(i);
    }

    level += 1;
  } // end while

  thandle.set_symbolic_complete();
  thandle.set_num_levels(level);

  // Create the chain now
  if ( thandle.algm_requires_symb_chain() ) {
    symbolic_chain_phase(thandle, nodes_per_level);
  }

  // Output check
#ifdef LVL_OUTPUT_INFO
  std::cout << "  set symbolic complete: " << thandle.is_symbolic_complete() << std::endl;
  std::cout << "  set num levels: " << thandle.get_num_levels() << std::endl;

  std::cout << "  upper_tri_symbolic result: " << std::endl;
  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "node: " << i << "  level_list = " << level_list(i) << std::endl; }

  for ( size_type i = 0; i < level; ++i )
  { std::cout << "level: " << i << "  nodes_per_level = " << nodes_per_level(i) << std::endl; }

  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "i: " << i << "  nodes_grouped_by_level = " << nodes_grouped_by_level(i) << std::endl; }
#endif

  // Deep copy to device views
  Kokkos::deep_copy(dnodes_grouped_by_level, nodes_grouped_by_level);
  Kokkos::deep_copy(dnodes_per_level, nodes_per_level);
  Kokkos::deep_copy(dlevel_list, level_list);
  Kokkos::deep_copy(diagonal_offsets, hdiagonal_offsets);
 }

#ifdef TRISOLVE_SYMB_TIMERS
 std::cout << "  Symbolic (upper tri) Total Time: " << timer_sym_uppertri_total.seconds() << std::endl;;
#endif
} // end upper_tri_symbolic







#ifdef DENSEPARTITION
template < class TriSolveHandle, class RowMapType, class EntriesType >
void symbolic_dense_partition_algm( TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries) {

#ifdef TRISOLVE_SYMB_TIMERS
  Kokkos::Timer timer_sym_dense_total;
#endif

  typedef typename TriSolveHandle::size_type size_type;
//  typedef typename TriSolveHandle::scalar_t  scalar_t;

  // FIXME - no longer need to run symbolic on shifted sparse graph, use the original where level scheduling know where to begin/end based on partitioning
  // This routine will:
  // *. Determine new offset into entries (and values), needed for partitioned rectspmtx block and dense trimtx
  // *. Allocate dense trimtx, and shifted row_map for rectangular spmtx block (likely needed for spmv)
  // *. Apply "shift" to rectspmtx row_map
  // *. Allocate shifted rectspmtx entries and values arrays (otherwise numbering is off for the spmv)
  // *. Call original symbolic routine
  // *.(TMP) Call numeric phase - will copy data into dense trimtx and rectspmtx ds
  // *.(TMP) To allocate rectspmtx ds, need the row_map allocated AND adjusted (i.e. dense trimtx colids removed) following some manipulation - make symbolic a "fused" symbolic+numeric, then use numeric to reload the data only (not realloc)


  // shifted rectspmtx row_map allocation
  auto dense_partition_nrows = thandle.get_dense_partition_nrows() ;

  thandle.alloc_row_map_rectspmtx(dense_partition_nrows+1);
  auto row_map_rectspmtx = thandle.get_row_map_rectspmtx();


  typedef Kokkos::View<size_type, typename RowMapType::memory_space> ShiftedEntriesStart;

  auto dense_row_start = thandle.get_dense_partition_row_start();

#ifdef DEBUGSYMBDENSE
  std::cout << "\n\n  Begin symbolic dense partition" << std::endl;
  std::cout << "  dense_row_start = " << dense_row_start << std::endl;
  std::cout << "  dense_nrows = " << dense_partition_nrows << std::endl;
  std::cout << "  drs + dense_nrows + 1 = " << dense_partition_nrows+dense_row_start+1 << std::endl;
#endif

  ShiftedEntriesStart shifted_entries_start(Kokkos::ViewAllocateWithoutInitializing("ses"));

  auto dprow_map = Kokkos::subview( drow_map, Kokkos::pair<size_type,size_type>(dense_row_start, dense_row_start+dense_partition_nrows+1) );


  auto h_shifted_entries_start_view = Kokkos::create_mirror_view(shifted_entries_start);
  Kokkos::deep_copy(h_shifted_entries_start_view, shifted_entries_start);
  // Use this to set offset into entries for copying to dense tri within numeric phase
  // TODO Is this even necessary now??? Can alternatively use original rowmap(dense_row_start) to point to beginning of entries and values to copy to dense tri...


  // reshape the rectspmtx row_map, allocate corresponding entries and vals
  //size_type rectsptmx_nnz = 0;
  size_type rectspmtx_col_start;
  size_type trimtx_col_start;

  bool is_lower_tri = thandle.is_lower_tri();

  if (is_lower_tri) {
    rectspmtx_col_start = 0; // ends at trimtx_col_start
    trimtx_col_start = dense_row_start; // ends at nrows
  }
  else {
    rectspmtx_col_start = dense_partition_nrows; // ends at nrows
    trimtx_col_start = 0; // ends at rectspmtx_col_start
  }

#ifdef DEBUGSYMBDENSE
  std::cout << "  is_lower_tri = " << is_lower_tri << std::endl;
  std::cout << "  rectspmtx_col_start = " << rectspmtx_col_start << std::endl;
  std::cout << "  trimtx_col_start = " << trimtx_col_start << std::endl;
#endif

  // reshape CRS
  // dprow_map has proper offsets into dentries and dvalues
  // Frequency count of indices in rectspmtx of partition
  Kokkos::parallel_for("row_map_rectspmtx freq count", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dense_partition_nrows),
    KOKKOS_LAMBDA (const size_type i) {
      size_type offset_start = dprow_map(i);
      size_type offset_end   = dprow_map(i+1);
      //printf("offset_start = %d  offset_end = %d\n", offset_start, offset_end);
      for (size_type offset = offset_start; offset < offset_end; ++offset) {
        size_type colid = dentries(offset);
        //printf("i = %d  colid = %d  is_lower_tri = %d\n",i, colid, (int)is_lower_tri);
        // Count in-sparse-rect entries per row, store at row_map(rowid+1) in anticipation of followup scan
        if ( (is_lower_tri && colid < trimtx_col_start) || (!is_lower_tri && colid >= rectspmtx_col_start) ) {
          ++row_map_rectspmtx(i+1);
        }
      }
    });
  Kokkos::fence();

#ifdef PRINT1DVIEWSSYMB
  std::cout << "  Freq count row_map_rectspmtx" << std::endl;
  print_view1d_symbolic(row_map_rectspmtx);
#endif

  // Convert count of indices in rectspmtx to a row_map
  size_type rectspmtx_nnz = 0; // this is needed for allocating rectspmtx entries and values arrays
  Kokkos::parallel_scan("row_map_rectspmtx scan", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, row_map_rectspmtx.extent(0)),
    KOKKOS_LAMBDA (const size_type i, size_type& update, const bool& final) {
      update += row_map_rectspmtx(i);
      if (final) {
        row_map_rectspmtx(i) = update;
      }
    }, rectspmtx_nnz);
  Kokkos::fence();

#ifdef PRINT1DVIEWSSYMB
  std::cout << "  Post-scan row_map_rectspmtx" << std::endl;
  print_view1d_symbolic(row_map_rectspmtx);
#endif

  thandle.set_nnz_rectspmtx(rectspmtx_nnz);

#ifdef DEBUGSYMBDENSE
  std::cout << "  rectspmtx_nnz = " << rectspmtx_nnz << std::endl;
#endif

  thandle.alloc_entries_rectspmtx(rectspmtx_nnz); 
  auto entries_rectspmtx= thandle.get_entries_rectspmtx(); 

  thandle.alloc_values_rectspmtx(rectspmtx_nnz);
  auto values_rectspmtx= thandle.get_values_rectspmtx();

  // create rectspmtx entries array
  Kokkos::parallel_for("entries_rectspmtx init", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dense_partition_nrows),
    KOKKOS_LAMBDA (const size_type i) {
      // Iterate over partition of original matrix to extract the indices in the rectangular sparse matrix partition
      size_type offset_start = dprow_map(i);
      size_type offset_end   = dprow_map(i+1);

      //auto idx_count_this_row = row_map_rectspmtx(i+1) - row_map_rectspmtx(i);
      auto new_idx_offset = row_map_rectspmtx(i);

      for (size_type offset = offset_start; offset < offset_end; ++offset) {
        size_type colid = dentries(offset);
        // Count in-sparse-rect entries per row, store at row_map(rowid+1) in anticipation of followup scan
        if ( (is_lower_tri && colid < trimtx_col_start) || (!is_lower_tri && colid >= rectspmtx_col_start) ) {
          entries_rectspmtx(new_idx_offset) = is_lower_tri ? colid : colid - dense_partition_nrows;
          ++new_idx_offset;
        }
      }

    });
  Kokkos::fence();

#ifdef PRINT1DVIEWSSYMB
  print_view1d_symbolic(entries_rectspmtx);
#endif
  
  // alloc dense tri mtx
  thandle.alloc_dense_trimtx(dense_partition_nrows, dense_partition_nrows);
  Kokkos::fence();
  auto dense_trimtx= thandle.get_dense_trimtx();

  auto sptrimtx_row_start = thandle.get_persist_sptrimtx_row_start();
  auto sptrimtx_nrows = thandle.get_persist_sptrimtx_nrows();
  auto sptrimtx_row_map = Kokkos::subview( drow_map, Kokkos::pair<size_type,size_type>(sptrimtx_row_start, sptrimtx_row_start+sptrimtx_nrows+1) );
  Kokkos::fence();

#ifdef DEBUGSYMBDENSE
  std::cout << " sptrimtx_row_start = " << sptrimtx_row_start << "  sptrimtx_nrows = " << sptrimtx_nrows << "  sptri end " << sptrimtx_row_start+sptrimtx_nrows+1 << std::endl;
  std::cout << "  Call lvl schedule symbolic" << std::endl;
#endif

  if (thandle.is_lower_tri()) {
    lower_tri_symbolic(thandle, sptrimtx_row_map, dentries);
  }
  else {
    upper_tri_symbolic(thandle, sptrimtx_row_map, dentries);
  }
#ifdef TRISOLVE_SYMB_TIMERS
  std::cout << "  Symbolic dense partition phase time: " << timer_sym_dense_total.seconds() << std::endl;;
#endif
}


template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType >
void numeric_dense_partition_algm(TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries, const ValuesType dvalues) {

#ifdef TRISOLVE_SYMB_TIMERS
  Kokkos::Timer timer_num_dense_total;
#endif
  typedef typename TriSolveHandle::size_type size_type;

  std::cout << "\n\n  Begin numeric" << std::endl;
  auto dense_partition_nrows = thandle.get_dense_partition_nrows() ;
  auto dense_row_start = thandle.get_dense_partition_row_start();

  auto dprow_map = Kokkos::subview( drow_map, Kokkos::pair<size_type,size_type>(dense_row_start, dense_row_start+dense_partition_nrows+1) );

  //size_type rectsptmx_nnz = 0;
  size_type rectspmtx_col_start;
  size_type trimtx_col_start;

  auto row_map_rectspmtx = thandle.get_row_map_rectspmtx();

  bool is_lower_tri = thandle.is_lower_tri();

  if (is_lower_tri) {
    rectspmtx_col_start = 0; // ends at trimtx_col_start
    trimtx_col_start = dense_row_start; // ends at nrows
  }
  else {
    rectspmtx_col_start = dense_partition_nrows; // ends at nrows
    trimtx_col_start = 0; // ends at rectspmtx_col_start
  }

#ifdef PRINT1DVIEWSSYMB
  std::cout << "  is_lower_tri = " << is_lower_tri << std::endl;
  std::cout << "  rectspmtx_col_start = " << rectspmtx_col_start << std::endl;
  std::cout << "  trimtx_col_start = " << trimtx_col_start << std::endl;
#endif

  auto dense_trimtx= thandle.get_dense_trimtx();

  auto values_rectspmtx= thandle.get_values_rectspmtx();

  // fill rectspmtx values array and dense trimtx
  Kokkos::parallel_for("numeric values fill init", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dense_partition_nrows),
    KOKKOS_LAMBDA (const size_type i) {
      // Iterate over partition of original matrix to extract the indices in the rectangular sparse matrix partition
      size_type offset_start = dprow_map(i);
      size_type offset_end   = dprow_map(i+1);

      //auto idx_count_this_row = row_map_rectspmtx(i+1) - row_map_rectspmtx(i);
      auto new_idx_offset = row_map_rectspmtx(i);

      for (size_type offset = offset_start; offset < offset_end; ++offset) {
        size_type colid = dentries(offset);
        auto val = dvalues(offset);
        // Count in-sparse-rect entries per row, store at row_map(rowid+1) in anticipation of followup scan
        if ( (is_lower_tri && colid < trimtx_col_start) || (!is_lower_tri && colid >= rectspmtx_col_start) ) {
          values_rectspmtx(new_idx_offset) = val;
          ++new_idx_offset;
        }
        else {
          auto trimtx_shifted_colid = is_lower_tri ? colid - trimtx_col_start : colid;
          dense_trimtx(i, trimtx_shifted_colid) = val;
        }
      }

    });
  Kokkos::fence();

#ifdef PRINT1DVIEWSSYMB
  print_view1d_symbolic(values_rectspmtx);
#endif

#ifdef PRINT1DVIEWSSYMB
  auto hdense_tri = Kokkos::create_mirror_view(dense_trimtx);
  Kokkos::deep_copy(hdense_tri, dense_trimtx);
  for (size_t i = 0; i < hdense_tri.extent(0); ++i) {
    for (size_t j = 0; j < hdense_tri.extent(1); ++j) {
      std::cout << "  hdense_tri(" << i << "," << j << ") = " << hdense_tri(i,j) << std::endl;
    }
  }
#endif

  auto diagonal_values = thandle.get_diagonal_values();
  auto diagonal_offsets = thandle.get_diagonal_offsets();
  if (diagonal_values.extent(0) != diagonal_offsets.extent(0)) {
    //std::cout << "ERROR: diagonal_values different size than diagonal_offsets" << std::endl;
    throw std::runtime_error ("  numeric error: diagonal_values different size than diagonal_offsets");
  }
  Kokkos::parallel_for("Store diagonal entries by rowid", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, diagonal_offsets.extent(0)), 
    KOKKOS_LAMBDA ( const size_type i ) {
      diagonal_values(i) = dvalues(diagonal_offsets(i));
    });
  Kokkos::fence();


  std::cout << "  numeric complete" << std::endl;

  thandle.set_numeric_complete();
#ifdef TRISOLVE_SYMB_TIMERS
  std::cout << "  Numeric Time: " << timer_num_dense_total.seconds() << std::endl;;
#endif
} // end numeric
#endif


} // namespace Experimental
} // namespace Impl
} // namespace KokkosSparse

#ifdef LVL_OUTPUT_INFO
#undef LVL_OUTPUT_INFO
#endif

#ifdef CHAIN_LVL_OUTPUT_INFO
#undef LVL_OUTPUT_INFO
#endif

#endif
