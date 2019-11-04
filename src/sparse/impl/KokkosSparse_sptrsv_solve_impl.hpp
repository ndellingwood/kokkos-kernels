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

#ifndef KOKKOSSPARSE_IMPL_SPTRSV_SOLVE_HPP_
#define KOKKOSSPARSE_IMPL_SPTRSV_SOLVE_HPP_

/// \file KokkosSparse_impl_sptrsv.hpp
/// \brief Implementation(s) of sparse triangular solve.

#include <KokkosKernels_config.h>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_sptrsv_handle.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

#include <KokkosBatched_Trsv_Decl.hpp>
#include <KokkosBatched_Trsv_Serial_Impl.hpp>

//#define TRISOLVE_TIMERS
//#define TRISOLVE_TIMERS_ITER_OUTPUT
//
//#define LVL_OUTPUT_INFO
//#define CHAIN_DEBUG_OUTPUT
//#define PRINT1DVIEWS
//#define SOLVE_DEBUG_OUTPUT
//#define USEDIAGVALUES
//#define SWAPPARFORS

#define SOLVE_IMPL_REFACTORING
//#define SERIAL_FOR_LOOP

#define TRILVLSCHED
#define LTCUDAGRAPHTEST

//#define KOKKOSPSTRSV_SOLVE_IMPL_PROFILE 1
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
#include "cuda_profiler_api.h"
#endif

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

//#ifdef DENSEPARTITION
#if defined(KOKKOS_ENABLED_CUDA)
#include "cuda_runtime.h"
#include "cublas_v2.h"
  #if defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION
  #define KOKKOSKERNELS_SPTRSV_CUDAGRAPHSUPPORT
  #endif
#endif

struct UnsortedTag {};

struct LargerCutoffTag {};

#ifdef PRINT1DVIEWS
template <class ViewType>
void print_view1d_solve(const ViewType dv) {
  auto v = Kokkos::create_mirror_view(dv);
  Kokkos::deep_copy(v, dv);
  std::cout << "Output for view " << v.label() << std::endl;
  for (size_t i = 0; i < v.extent(0); ++i) {
    std::cout << "v(" << i << ") = " << v(i) << " , ";
  }
  std::cout << std::endl;
}
#endif


// This functor unifies the lower and upper implementations, the hope is the "is_lowertri" check does not add noticable time on larger problems
template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct TriLvlSchedTP1SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  const bool is_lowertri;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long dense_nrows;


  TriLvlSchedTP1SolverFunctor(const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, const bool is_lowertri_, long node_count_, long dense_nrows_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), is_lowertri(is_lowertri_), node_count(node_count_), dense_nrows(dense_nrows_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_rank = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

      Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          // FIXME: This check for col != row vs col > row will be broken with partial persist_sptrimtx, since using a row_map with rowid shifted to 0 but not colid

   // FIXME Need to pass dense_rows to these functors...
#ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif

          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
      }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_rank == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = is_lowertri ? (rhs_rowid+diff)/values(eoffset-1) : (rhs_rowid+diff)/values(soffset);
        }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_rank = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        auto diag = -1;

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
          else {
            diag = ptr;
          }
        }, diff );
        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_rank == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = (rhs_rowid+diff)/values(diag);
        }
  }
};



template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct TriLvlSchedTP1SolverFunctorDiagValues
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;
  ValuesType diagonal_values; // inserted according to rowid

  const bool is_lowertri;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long dense_nrows;


  TriLvlSchedTP1SolverFunctorDiagValues( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, const ValuesType &diagonal_values_, const bool is_lowertri_, long node_count_, long dense_nrows_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), diagonal_values(diagonal_values_), is_lowertri(is_lowertri_), node_count(node_count_), dense_nrows(dense_nrows_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_rank = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

      Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          // FIXME: This check for col != row vs col > row will be broken with partial persist_sptrimtx, since using a row_map with rowid shifted to 0 but not colid

   // FIXME Need to pass dense_rows to these functors...
#ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif

          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
      }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_rank == 0 )
        {
          //lhs(rowid) = is_lowertri ? (rhs_rowid+diff)/values(eoffset-1) : (rhs_rowid+diff)/values(soffset);
          lhs(rowid) = (rhs_rowid+diff)/diagonal_values(rowid);
        }
  }

};


template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct TriLvlSchedTP2SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  const bool is_lowertri;
  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long node_groups;
  long dense_nrows;


  TriLvlSchedTP2SolverFunctor(const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, const bool is_lowertri_, long node_count_, long node_groups_ = 0, long dense_nrows_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), is_lowertri(is_lowertri_), node_count(node_count_), node_groups(node_groups_), dense_nrows(dense_nrows_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {

            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              //auto colid = entries(ptr);

#ifdef DENSEPARTITION
              auto original_col = entries(ptr);
              //auto colid = original_col - dense_nrows; //shift required for upper-tri
              auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
#else
              auto colid = entries(ptr);
#endif

              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at eoffset - 1
            lhs(rowid) = is_lowertri ? (rhs_rowid+diff)/values(eoffset-1) : (rhs_rowid+diff)/values(soffset);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {
            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            auto diag = -1;
            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              //auto colid = entries(ptr);
#ifdef DENSEPARTITION
              auto original_col = entries(ptr);
              //auto colid = original_col - dense_nrows; //shift required for upper-tri
              auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
#else
              auto colid = entries(ptr);
#endif
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
              else {
                diag = ptr;
              }
            }, diff );

            lhs(rowid) = (rhs_rowid+diff)/values(diag);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }
};


// Lower vs Upper Multi-block Functors

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedRPSolverFunctor
{
  typedef typename EntriesType::non_const_value_type lno_t;
  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  LowerTriLvlSchedRPSolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, NGBLType nodes_grouped_by_level_ ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    // Assuming indices are sorted per row, diag entry is final index in the list

    auto soffset = row_map(rowid);
    auto eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);

    for ( auto ptr = soffset; ptr < eoffset; ++ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        lhs(rowid) = rhs_rowid/val;
      }
    } // end for ptr
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    auto soffset = row_map(rowid);
    auto eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    auto diag = -1;

    for ( auto ptr = soffset; ptr < eoffset; ++ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        diag = ptr;
      }
    } // end for ptr
    lhs(rowid) = rhs_rowid/values(diag);
  }
};



template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedTP1SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long node_groups;


  LowerTriLvlSchedTP1SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_rank = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

      Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
      }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_rank == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = (rhs_rowid+diff)/values(eoffset-1);
        }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_rank = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        auto diag = -1;

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
          else {
            diag = ptr;
          }
        }, diff );
        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this; can also use Kokkos::single
        if ( my_rank == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
          lhs(rowid) = (rhs_rowid+diff)/values(diag);
        }
  }
};


// FIXME CUDA: This algorithm not working with all integral type combos
// In any case, this serves as a skeleton for 3-level hierarchical parallelism for alg dev
template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedTP2SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long node_groups;


  LowerTriLvlSchedTP2SolverFunctor(const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {

            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at eoffset - 1
            lhs(rowid) = (rhs_rowid+diff)/values(eoffset-1);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {
            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            auto diag = -1;
            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
              else {
                diag = ptr;
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at eoffset - 1
            lhs(rowid) = (rhs_rowid+diff)/values(diag);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }
};

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedRPSolverFunctor
{
  typedef typename EntriesType::non_const_value_type lno_t;
  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;


  UpperTriLvlSchedRPSolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_ ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    // Assuming indices are sorted per row, diag entry is final index in the list
    long soffset = row_map(rowid);
    long eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    for ( long ptr = eoffset-1; ptr >= soffset; --ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        lhs(rowid) = rhs_rowid/val;
      }
    } // end for ptr
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const lno_t i) const {
    auto rowid = nodes_grouped_by_level(i);
    long soffset = row_map(rowid);
    long eoffset = row_map(rowid+1);
    auto rhs_rowid = rhs(rowid);
    auto diag = -1;
    for ( long ptr = eoffset-1; ptr >= soffset; --ptr ) {
      auto colid = entries(ptr);
      auto val   = values(ptr);
      if ( colid != rowid ) {
        rhs_rowid = rhs_rowid - val*lhs(colid);
      }
      else {
        diag = ptr;
      }
    } // end for ptr
    lhs(rowid) = rhs_rowid/values(diag);
  }

};


template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedTP1SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long node_groups;


  UpperTriLvlSchedTP1SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_rank = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
        }, diff );

        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this, also can use Kokkos::single
        if ( my_rank == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at start offset
          lhs(rowid) = (rhs_rowid+diff)/values(soffset);
        }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid
        auto rowid = nodes_grouped_by_level(my_league + node_count);
        auto my_rank = team.team_rank();

        auto soffset = row_map(rowid);
        auto eoffset = row_map(rowid+1);
        auto rhs_rowid = rhs(rowid);
        scalar_t diff = scalar_t(0.0);

        auto diag = -1;

        Kokkos::parallel_reduce( Kokkos::TeamThreadRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
          if ( colid != rowid ) {
            tdiff = tdiff - val*lhs(colid);
          }
          else {
            diag = ptr;
          }
        }, diff );
        team.team_barrier();

        // At end, finalize rowid == colid
        // only one thread should do this, also can use Kokkos::single
        if ( my_rank == 0 )
        {
        // ASSUMPTION: sorted diagonal value located at start offset
          lhs(rowid) = (rhs_rowid+diff)/values(diag);
        }
  }
};


// FIXME CUDA: This algorithm not working with all integral type combos
// In any case, this serves as a skeleton for 3-level hierarchical parallelism for alg dev
template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedTP2SolverFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long node_groups;


  UpperTriLvlSchedTP2SolverFunctor(const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, long node_count_, long node_groups_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), node_count(node_count_), node_groups(node_groups_) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & team) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {

            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at start offset
            lhs(rowid) = (rhs_rowid+diff)/values(soffset);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const UnsortedTag&, const member_type & team ) const {
        auto my_league = team.league_rank(); // map to rowid

        size_t nrows = row_map.extent(0) - 1;

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, 0, node_groups ), [&] ( const long ng ) {
          auto rowid = nodes_grouped_by_level(node_count + my_league*node_groups + ng);
          if ( size_t(rowid) < nrows ) {
            auto soffset = row_map(rowid);
            auto eoffset = row_map(rowid+1);
            auto rhs_rowid = rhs(rowid);
            scalar_t diff = scalar_t(0.0);

            auto diag = -1;
            Kokkos::parallel_reduce( Kokkos::ThreadVectorRange( team, soffset, eoffset ), [&] ( const long ptr, scalar_t &tdiff ) {
              auto colid = entries(ptr);
              auto val   = values(ptr);
              if ( colid != rowid ) {
                tdiff = tdiff - val*lhs(colid);
              }
              else {
                diag = ptr;
              }
            }, diff );

            // ASSUMPTION: sorted diagonal value located at start offset
            lhs(rowid) = (rhs_rowid+diff)/values(diag);
          } // end if
        }); // end TeamThreadRange

        team.team_barrier();
  }

};


// --------------------------------
// Single-block functors
// --------------------------------

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct LowerTriLvlSchedTP1SingleBlockFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;
  NGBLType nodes_per_level;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long lvl_start;
  long lvl_end;
  long cutoff;
  // team_size: each team can be assigned a row, if there are enough rows...


  LowerTriLvlSchedTP1SingleBlockFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, NGBLType &nodes_per_level_, long node_count_, long lvl_start_, long lvl_end_, long cutoff_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), nodes_per_level(nodes_per_level_), node_count(node_count_), lvl_start(lvl_start_), lvl_end(lvl_end_), cutoff(cutoff_) {}

  // SingleBlock: Only one block (or league) executing; team_rank used to map thread to row

  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      diff = scalar_t(0.0);

      if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is acting over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with the TeamVectorRange
// Round 2: Use TeamVectorRange Policy

#ifdef SERIAL_FOR_LOOP
        for (auto ptr = soffset; ptr < eoffset; ++ptr) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
  #ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
  #endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
#else
        auto trange = eoffset - soffset;
        // NOTE TeamVectorRange flattens teamthread + threadvector ranges
        //Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        {
          auto ptr = soffset + loffset;

          auto colid = entries(ptr);
          auto val   = values(ptr);
  #ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
  #endif
          if ( colid != rowid ) {
            tdiff -= val*lhs(colid);
          }
        }, diff);
        team.team_barrier();
#endif
        // ASSUMPTION: sorted diagonal value located at eoffset - 1
        lhs(rowid) = (rhs_val+diff)/values(eoffset-1);
        // else if uppertri
        //   lhs(rowid) = (rhs_val+diff)/values(soffset);

      } // end if team.team_rank() < nodes_this_lvl
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end operator


  KOKKOS_INLINE_FUNCTION
  void operator()( const LargerCutoffTag&, const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_team_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      // If cutoff > team_size, then a thread will be responsible for multiple rows - this may be a helpful scenario depending on occupancy etc.
      for (int my_rank = my_team_rank; my_rank < cutoff; my_rank+=team.team_size() ) {
      diff = scalar_t(0.0);
       if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is coordinating over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with a for loop, then TeamVectorRange
// Round 2: Use TeamVectorRange Policy

#ifdef SERIAL_FOR_LOOP
        for (auto ptr = soffset; ptr < eoffset; ++ptr) {
#ifdef DENSEPARTITIONL
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif
          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
#else
        auto trange = eoffset - soffset;
        // NOTE TeamVectorRange flattens teamthread + threadvector ranges
        //Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        {
          auto ptr = soffset + loffset;
#ifdef DENSEPARTITIONL
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif
          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            tdiff -= val*lhs(colid);
          }
        },diff);
        team.team_barrier();
#endif
        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
          lhs(rowid) = (rhs_val+diff)/values(eoffset-1);
       } // end if team.team_rank() < nodes_this_lvl
      } // end for my_rank loop
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end tagged operator

};

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct UpperTriLvlSchedTP1SingleBlockFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;
  NGBLType nodes_per_level;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long lvl_start;
  long lvl_end;
  long cutoff;
  // team_size: each team can be assigned a row, if there are enough rows...


  UpperTriLvlSchedTP1SingleBlockFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, NGBLType &nodes_per_level_, long node_count_, long lvl_start_, long lvl_end_, long cutoff_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), nodes_per_level(nodes_per_level_), node_count(node_count_), lvl_start(lvl_start_), lvl_end(lvl_end_), cutoff(cutoff_) {}

  // SingleBlock: Only one block (or league) executing; team_rank used to map thread to row

  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      diff = scalar_t(0.0);

      if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is acting over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with the TeamVectorRange
// Round 2: Use TeamVectorRange Policy

#ifdef SERIAL_FOR_LOOP
        for (auto ptr = soffset; ptr < eoffset; ++ptr) {
          auto colid = entries(ptr);
          auto val   = values(ptr);
  #ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
  #endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
#else
        auto trange = eoffset - soffset;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        {
          auto ptr = soffset + loffset;
          auto colid = entries(ptr);
          auto val   = values(ptr);
  #ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
  #endif
          if ( colid != rowid ) {
            tdiff -= val*lhs(colid);
          }
        }, diff);
        team.team_barrier();
#endif
        // ASSUMPTION: sorted diagonal value located at soffset
        lhs(rowid) = (rhs_val+diff)/values(soffset);

      } // end if
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl each thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end operator


  KOKKOS_INLINE_FUNCTION
  void operator()( const LargerCutoffTag&, const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_team_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      // If cutoff > team_size, then a thread will be responsible for multiple rows - this may be a helpful scenario depending on occupancy etc.
      for (int my_rank = my_team_rank; my_rank < cutoff; my_rank+=team.team_size() ) {
      diff = scalar_t(0.0);
       if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is coordinating over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with a for loop, then TeamVectorRange
// Round 2: Use TeamVectorRange Policy

#ifdef SERIAL_FOR_LOOP
        for (auto ptr = soffset; ptr < eoffset; ++ptr) {
#ifdef DENSEPARTITIONU
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif
          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
#else
        auto trange = eoffset - soffset;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        {
          auto ptr = soffset + loffset;
#ifdef DENSEPARTITIONU
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif
          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            tdiff -= val*lhs(colid);
          }
        }, diff);
        team.team_barrier();
#endif
        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
          lhs(rowid) = (rhs_val+diff)/values(soffset);
       } // end if team.team_rank() < nodes_this_lvl
      } // end for my_rank loop
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end tagged operator
};



template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct TriLvlSchedTP1SingleBlockFunctor
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;
  NGBLType nodes_per_level;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long lvl_start;
  long lvl_end;
  const bool is_lowertri;
  const int dense_nrows;
  const int  cutoff;
  // team_size: each team can be assigned a row, if there are enough rows...


  TriLvlSchedTP1SingleBlockFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, NGBLType &nodes_per_level_, long node_count_, long lvl_start_, long lvl_end_, const bool is_lower_, const int dense_nrows_ = 0, const int cutoff_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), nodes_per_level(nodes_per_level_), node_count(node_count_), lvl_start(lvl_start_), lvl_end(lvl_end_), is_lowertri(is_lower_), dense_nrows(dense_nrows_), cutoff(cutoff_) {}

  // SingleBlock: Only one block (or league) executing; team_rank used to map thread to row

  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      diff = scalar_t(0.0);

      if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is coordinating over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with a for loop, then TeamVectorRange
// Round 2: Use TeamVectorRange Policy
#ifdef SERIAL_FOR_LOOP
        for (auto ptr = soffset; ptr < eoffset; ++ptr) {

       // FIXME Need to pass dense_rows to these functors...
  #ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
  #else
          auto colid = entries(ptr);
  #endif

          auto val   = values(ptr);
  #ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
  #endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
#else
        auto trange = eoffset - soffset;
        // NOTE TeamVectorRange flattens teamthread + threadvector ranges
        //Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        {
          auto ptr = soffset + loffset;
  #ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
  #else
          auto colid = entries(ptr);
  #endif

          auto val   = values(ptr);
  #ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
  #endif
          if ( colid != rowid ) {
            tdiff -= val*lhs(colid);
          }
        }, diff);
      team.team_barrier();
#endif

        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
        if (is_lowertri)
          lhs(rowid) = (rhs_val+diff)/values(eoffset-1);
        else
          lhs(rowid) = (rhs_val+diff)/values(soffset);
        // else if uppertri
        //   lhs(rowid) = (rhs_val+diff)/values(soffset);

      } // end if team.team_rank() < nodes_this_lvl
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end operator


  KOKKOS_INLINE_FUNCTION
  void operator()( const LargerCutoffTag&, const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_team_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      // If cutoff > team_size, then a thread will be responsible for multiple rows - this may be a helpful scenario depending on occupancy etc.
      for (int my_rank = my_team_rank; my_rank < cutoff; my_rank+=team.team_size() ) {
      diff = scalar_t(0.0);
       if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is coordinating over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with a for loop, then TeamVectorRange
// Round 2: Use TeamVectorRange Policy

#ifdef SERIAL_FOR_LOOP
        for (auto ptr = soffset; ptr < eoffset; ++ptr) {
  #ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
  #else
          auto colid = entries(ptr);
  #endif
          auto val   = values(ptr);
  #ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
  #endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
#else
        auto trange = eoffset - soffset;
        // NOTE TeamVectorRange flattens teamthread + threadvector ranges
        //Kokkos::parallel_reduce(Kokkos::TeamVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, trange), [&] (const int loffset, scalar_t& tdiff)
        {
          auto ptr = soffset + loffset;
  #ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
  #else
          auto colid = entries(ptr);
  #endif
          auto val   = values(ptr);
  #ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
  #endif
          if ( colid != rowid ) {
            tdiff -= val*lhs(colid);
          }
        }, diff);

        team.team_barrier();
#endif

        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
        if (is_lowertri)
          lhs(rowid) = (rhs_val+diff)/values(eoffset-1);
        else
          lhs(rowid) = (rhs_val+diff)/values(soffset);
        // else if uppertri
        //   lhs(rowid) = (rhs_val+diff)/values(soffset);

       } // end if team.team_rank() < nodes_this_lvl
      } // end for my_rank loop
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end tagged operator

};


template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct TriLvlSchedTP1SingleBlockFunctor_Backup
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;
  NGBLType nodes_per_level;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long lvl_start;
  long lvl_end;
  const bool is_lowertri;
  const int dense_nrows;
  const int  cutoff;
  // team_size: each team can be assigned a row, if there are enough rows...


  TriLvlSchedTP1SingleBlockFunctor_Backup( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, NGBLType &nodes_per_level_, long node_count_, long lvl_start_, long lvl_end_, const bool is_lower_, const int dense_nrows_ = 0, const int cutoff_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), nodes_per_level(nodes_per_level_), node_count(node_count_), lvl_start(lvl_start_), lvl_end(lvl_end_), is_lowertri(is_lower_), dense_nrows(dense_nrows_), cutoff(cutoff_) {}

  // SingleBlock: Only one block (or league) executing; team_rank used to map thread to row

  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      diff = scalar_t(0.0);

      if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is coordinating over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with a for loop, then TeamVectorRange
// Round 2: Use TeamVectorRange Policy

        for (auto ptr = soffset; ptr < eoffset; ++ptr) {

   // FIXME Need to pass dense_rows to these functors...
#ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif

          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
        if (is_lowertri)
          lhs(rowid) = (rhs_val+diff)/values(eoffset-1);
        else
          lhs(rowid) = (rhs_val+diff)/values(soffset);
        // else if uppertri
        //   lhs(rowid) = (rhs_val+diff)/values(soffset);

      } // end if team.team_rank() < nodes_this_lvl
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end operator


  KOKKOS_INLINE_FUNCTION
  void operator()( const LargerCutoffTag&, const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_team_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      // If cutoff > team_size, then a thread will be responsible for multiple rows - this may be a helpful scenario depending on occupancy etc.
      for (int my_rank = my_team_rank; my_rank < cutoff; my_rank+=team.team_size() ) {
      diff = scalar_t(0.0);
       if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is coordinating over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with a for loop, then TeamVectorRange
// Round 2: Use TeamVectorRange Policy

        for (auto ptr = soffset; ptr < eoffset; ++ptr) {
#ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif
          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
        if (is_lowertri)
          lhs(rowid) = (rhs_val+diff)/values(eoffset-1);
        else
          lhs(rowid) = (rhs_val+diff)/values(soffset);
        // else if uppertri
        //   lhs(rowid) = (rhs_val+diff)/values(soffset);

       } // end if team.team_rank() < nodes_this_lvl
      } // end for my_rank loop
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end tagged operator

};

template <class RowMapType, class EntriesType, class ValuesType, class LHSType, class RHSType, class NGBLType>
struct TriLvlSchedTP1SingleBlockFunctorDiagValues
{
  typedef typename RowMapType::execution_space execution_space;
  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  typedef typename policy_type::member_type member_type;
  typedef typename EntriesType::non_const_value_type lno_t;
  typedef typename ValuesType::non_const_value_type scalar_t;

  RowMapType row_map;
  EntriesType entries;
  ValuesType values;
  LHSType lhs;
  RHSType rhs;
  NGBLType nodes_grouped_by_level;
  NGBLType nodes_per_level;
  ValuesType diagonal_values;

  long node_count; // like "block" offset into ngbl, my_league is the "local" offset
  long lvl_start;
  long lvl_end;
  const bool is_lowertri;
  const int dense_nrows;
  const int  cutoff;
  // team_size: each team can be assigned a row, if there are enough rows...


  TriLvlSchedTP1SingleBlockFunctorDiagValues( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, const NGBLType &nodes_per_level_, const ValuesType &diagonal_values_, long node_count_, const long lvl_start_, const long lvl_end_, const bool is_lower_, const int dense_nrows_ = 0, const int cutoff_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), nodes_per_level(nodes_per_level_), diagonal_values(diagonal_values_), node_count(node_count_), lvl_start(lvl_start_), lvl_end(lvl_end_), is_lowertri(is_lower_), dense_nrows(dense_nrows_), cutoff(cutoff_) {}

  // SingleBlock: Only one block (or league) executing; team_rank used to map thread to row

  KOKKOS_INLINE_FUNCTION
  void operator()( const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      diff = scalar_t(0.0);

      if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is coordinating over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with a for loop, then TeamVectorRange
// Round 2: Use TeamVectorRange Policy

        for (auto ptr = soffset; ptr < eoffset; ++ptr) {

   // FIXME Need to pass dense_rows to these functors...
#ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif

          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
        lhs(rowid) = (rhs_val+diff)/diagonal_values(rowid);
      } // end if team.team_rank() < nodes_this_lvl
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end operator


  KOKKOS_INLINE_FUNCTION
  void operator()( const LargerCutoffTag&, const member_type & team ) const {
    long mut_node_count = node_count;
      typename NGBLType::non_const_value_type rowid {0};
      typename RowMapType::non_const_value_type soffset {0};
      typename RowMapType::non_const_value_type eoffset {0};
      typename RHSType::non_const_value_type rhs_val {0};
      scalar_t diff = scalar_t(0.0);
    for ( auto lvl = lvl_start; lvl < lvl_end; ++lvl ) {
      auto nodes_this_lvl = nodes_per_level(lvl);
      int my_team_rank = team.team_rank();
#ifdef CHAIN_DEBUG_OUTPUT
      printf("league_rank: %d  team_rank: %d  lvl: %ld  nodes_this_lvl: %ld\n", team.league_rank(), team.team_rank(), lvl, (long)nodes_this_lvl);
#endif
      // If cutoff > team_size, then a thread will be responsible for multiple rows - this may be a helpful scenario depending on occupancy etc.
      for (int my_rank = my_team_rank; my_rank < cutoff; my_rank+=team.team_size() ) {
       diff = scalar_t(0.0);
       if (my_rank < nodes_this_lvl) {

        // THIS is where the mapping of threadid to rowid happens
        rowid = nodes_grouped_by_level(my_rank + mut_node_count);

        soffset = row_map(rowid);
        eoffset = row_map(rowid+1);
        rhs_val = rhs(rowid);
#ifdef CHAIN_DEBUG_OUTPUT
        printf("league_rank: %d  team_rank: %d  node_count: %ld  lvl_start: %ld  lvl_end: %ld\n", team.league_rank(), team.team_rank(), mut_node_count, lvl_start, lvl_end);
        printf("  team_rank: %d  mut_node_count: %ld  ngbl: %ld\n", team.team_rank(), mut_node_count, (long)rowid);
        printf("  team_rank passed if: %d  rowid: %d  soffset: %d  eoffset: %d  rhs_val: %lf\n", team.team_rank(), (int)rowid, (int)soffset, (int)eoffset, rhs_val);
#endif

// FIXME NOTES:
// Assumptions: 1. Each "thread" owns a row in the level
//              2. At this point, the nested parallel_reduce is coordinating over all threads within the team!!!!! This is the bug in nt > 1 case.
// FIX:
//              Replace TeamThreadRange (which is allowing threads to cooperate over the solve...) with a for loop, then TeamVectorRange
// Round 2: Use TeamVectorRange Policy

        for (auto ptr = soffset; ptr < eoffset; ++ptr) {
#ifdef DENSEPARTITION
          auto original_col = entries(ptr);
          auto colid = is_lowertri ? original_col : original_col - dense_nrows; //shift required for upper-tri
          //auto colid = original_col - dense_nrows; //shift required for upper-tri
#else
          auto colid = entries(ptr);
#endif
          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
        lhs(rowid) = (rhs_val+diff)/diagonal_values(rowid);
       } // end if team.team_rank() < nodes_this_lvl
      } // end for my_rank loop
      {
        // Update mut_node_count from nodes_per_level(lvl) each iteration of lvl per thread
        mut_node_count += nodes_this_lvl;
      }
      team.team_barrier();
    } // end for lvl
  } // end tagged operator

};



// --------------------------------
// solver control routines
// --------------------------------


// TODO To make this work:
// - Create cum_nodes_per_level array via parallel_scan
// - Remove different solver versions to eliminate if-else branches of kernel calls
// - Modify functors to take the cum_nodes_per_level array, needed to map thread-block to work item
//

#if defined(LTCUDAGRAPHTEST) && defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION

template <class SpaceType>
struct ReturnTeamPolicyType;

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct ReturnTeamPolicyType<Kokkos::Serial> {
  using PolicyType = Kokkos::TeamPolicy<Kokkos::Serial>;

  static inline
  PolicyType get_policy(int nt, int ts) {
    return PolicyType(nt,ts);
  }

  template <class ExecInstanceType>
  static inline
  PolicyType get_policy(int nt, int ts, ExecInstanceType ) {
    return PolicyType(nt,ts);
    //return PolicyType(ExecInstanceType(),nt,ts);
  }
};
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct ReturnTeamPolicyType<Kokkos::OpenMP> {
  using PolicyType = Kokkos::TeamPolicy<Kokkos::OpenMP>;

  static inline
  PolicyType get_policy(int nt, int ts) {
    return PolicyType(nt,ts);
  }

  template <class ExecInstanceType>
  static inline
  PolicyType get_policy(int nt, int ts, ExecInstanceType ) {
    return PolicyType(nt,ts);
    //return PolicyType(ExecInstanceType(),nt,ts);
  }
};
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct ReturnTeamPolicyType<Kokkos::Cuda> {
  using PolicyType = Kokkos::TeamPolicy<Kokkos::Cuda>;

  static inline
  PolicyType get_policy(int nt, int ts) {
    return PolicyType(nt,ts);
  }

  template <class ExecInstanceType>
  static inline
  PolicyType get_policy(int nt, int ts, ExecInstanceType stream) {
    return PolicyType(stream,nt,ts);
  }
};
#endif

template <class SpaceType>
struct ReturnRangePolicyType;

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct ReturnRangePolicyType<Kokkos::Serial> {
  using PolicyType = Kokkos::RangePolicy<Kokkos::Serial>;

  static inline
  PolicyType get_policy(int nt, int ts) {
    return PolicyType(nt,ts);
  }

  template <class ExecInstanceType>
  static inline
  PolicyType get_policy(int nt, int ts, ExecInstanceType ) {
    return PolicyType(nt,ts);
    //return PolicyType(ExecInstanceType(),nt,ts);
  }
};
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct ReturnRangePolicyType<Kokkos::OpenMP> {
  using PolicyType = Kokkos::RangePolicy<Kokkos::OpenMP>;

  static inline
  PolicyType get_policy(int nt, int ts) {
    return PolicyType(nt,ts);
  }

  template <class ExecInstanceType>
  static inline
  PolicyType get_policy(int nt, int ts, ExecInstanceType ) {
    return PolicyType(nt,ts);
    //return PolicyType(ExecInstanceType(),nt,ts);
  }
};
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct ReturnRangePolicyType<Kokkos::Cuda> {
  using PolicyType = Kokkos::RangePolicy<Kokkos::Cuda>;

  static inline
  PolicyType get_policy(int nt, int ts) {
    return PolicyType(nt,ts);
  }

  template <class ExecInstanceType>
  static inline
  PolicyType get_policy(int nt, int ts, ExecInstanceType stream) {
    return PolicyType(stream,nt,ts);
  }
};
#endif

template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void lower_tri_solve_cg( TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs) {

#if 1
{
    typename TriSolveHandle::SPTRSVcudaGraphWrapperType* lcl_cudagraph = thandle.get_sptrsvCudaGraph();

    auto nlevels = thandle.get_num_levels();
    std::cout << "Begin Solve: nlevels = " << nlevels << std::endl;

    auto stream1 = lcl_cudagraph->stream;
    Kokkos::Cuda cuda1(stream1);
    auto graph = lcl_cudagraph->cudagraph;

    //cudaStream_t stream1;
    //cudaGraph_t graph;

    // Create a stream
    //cudaStreamCreate(&stream1);
   // Kokkos::Cuda cuda1(stream1);

// WIthout this, funky error...
#if 0
    Kokkos::parallel_for("Init",1,KOKKOS_LAMBDA (const int i) {
    });

    //Kokkos::fence();
#else
    Kokkos::Cuda().fence();
//    Kokkos::fence();
    cudaStreamSynchronize(stream1);
//    Kokkos::fence();
#endif

    typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;
    typedef typename TriSolveHandle::execution_space execution_space;
    typedef typename TriSolveHandle::size_type size_type;
    auto hnodes_per_level = thandle.get_host_nodes_per_level();
    auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

    size_type node_count = 0;

    //typedef Kokkos::TeamPolicy<execution_space> policy_type;
    int team_size = thandle.get_team_size();
    team_size = team_size == -1 ? 64 : team_size;

    //auto graphExec = lcl_cudagraph->cudagraphinstance;
    // Start capturing stream
    if(thandle.cudagraphCreated == false) {
    Kokkos::fence();
    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
    {
      for (int iter = 0; iter < nlevels; ++iter) {
        size_type lvl_nodes = hnodes_per_level(iter);

        using policy_type = ReturnTeamPolicyType<execution_space>;

        Kokkos::parallel_for("parfor_l_team_cudagraph",  Kokkos::Experimental::require(ReturnTeamPolicyType<execution_space>::get_policy(lvl_nodes,team_size,cuda1), Kokkos::Experimental::WorkItemProperty::HintLightWeight), LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count));

        node_count += hnodes_per_level(iter);
      }
    }
    cudaStreamEndCapture(stream1, &graph);

    // Create graphExec
    //cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphInstantiate(&(lcl_cudagraph->cudagraphinstance), graph, NULL, NULL, 0);
      thandle.cudagraphCreated = true;
    }
    // Run graph
    Kokkos::fence();
    //cudaGraphLaunch(graphExec, stream1);
    cudaGraphLaunch(lcl_cudagraph->cudagraphinstance, stream1);

    cudaStreamSynchronize(stream1);
    Kokkos::fence();
    std::cout << "  Kokkos stream example check" << std::endl;
//  Kokkos::abort("Exit");
}




#elif defined(CHRISTIANSTESTEXAMPLE)
// Christian's example - https://github.com/kokkos/kokkos/issues/2132 
{
    typename TriSolveHandle::SPTRSVcudaGraphWrapperType* lcl_cudagraph = thandle.get_sptrsvCudaGraph();


    typedef Kokkos::CudaSpace result_space_type;

    auto nlevels = thandle.get_num_levels();
    int N = 10;
    int R = 1;
    Kokkos::View<int*> a("A",N);

    auto stream1 = lcl_cudagraph->stream;
    Kokkos::Cuda cuda1(stream1);
    auto graph = lcl_cudagraph->cudagraph;

    //cudaStream_t stream1;
    //cudaGraph_t graph;

    // Create a stream
    //cudaStreamCreate(&stream1);
   // Kokkos::Cuda cuda1(stream1);


    Kokkos::parallel_for("Init",N,KOKKOS_LAMBDA (const int i) {
      a(i) = i;
    });

    Kokkos::View<int64_t, result_space_type> result("Result");

    Kokkos::fence();

      Kokkos::parallel_for("Add-5",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i) {
        a(i) += 5;
      });

      Kokkos::parallel_for("Sub-3",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i) {
       a(i) -= 3;
      });

      Kokkos::parallel_reduce("reduce",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i, int64_t& lsum) {
       lsum += a(i);
      },result);
      cudaStreamSynchronize(stream1);

    Kokkos::parallel_for("Init",N,KOKKOS_LAMBDA (const int i) {
      a(i) = i;
    });
    Kokkos::fence();

    Kokkos::Timer timer;
    for(int r=0; r<R; r++) {
      Kokkos::parallel_for("Add-5",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i) {
        a(i) += 5;
      });

      Kokkos::parallel_for("Sub-3",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i) {
       a(i) -= 3;
      });

      Kokkos::parallel_reduce("reduce",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i, int64_t& lsum) {
       lsum += a(i);
      },result);
      int64_t s_result;
      if(std::is_same<result_space_type,Kokkos::CudaSpace>::value) {
        Kokkos::deep_copy(cuda1,s_result,result);
        cudaStreamSynchronize(stream1);
      } else {
        cudaStreamSynchronize(stream1);
        s_result = result();
      }
      printf("Result %i %li %li\n",r,s_result,int64_t(N)*int64_t(N-1)/2+int64_t(N)*2*(r+1));
    }
    double time = timer.seconds();
    printf("NoGraphTime<%s>: %e\n",result_space_type::name(),time);

    Kokkos::parallel_for("Init",N,KOKKOS_LAMBDA (const int i) {
      a(i) = i;
    });

    typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;
    typedef typename TriSolveHandle::execution_space execution_space;
    typedef typename TriSolveHandle::size_type size_type;
    auto hnodes_per_level = thandle.get_host_nodes_per_level();
    auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

    size_type node_count = 0;

    typedef Kokkos::TeamPolicy<execution_space> policy_type;
    int team_size = thandle.get_team_size();
    team_size = team_size == -1 ? 64 : team_size;

    Kokkos::fence();
    // Start capturing stream
    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
    {
      for (int iter = 0; iter < nlevels; ++iter) {
      size_type lvl_nodes = hnodes_per_level(iter);
      Kokkos::parallel_for("Add-5",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i) {
        a(i) += 5;
      });

      Kokkos::parallel_for("Sub-3",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i) {
       a(i) -= 3;
      });

      Kokkos::parallel_reduce("reduce",Kokkos::RangePolicy<>(cuda1,0,N),KOKKOS_LAMBDA (const int i, int64_t& lsum) {
       lsum += a(i);
      },result);

      Kokkos::parallel_for("TP",Kokkos::TeamPolicy<>(cuda1,lvl_nodes,32),KOKKOS_LAMBDA (const typename Kokkos::TeamPolicy<>::member_type & member) {
        if (member.league_rank() == 0 && member.team_rank()==0)
          a(0) -= 3;

        member.team_barrier();

        double dresult = nlevels;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member,0,3), [=] (const int i, double &update) {
          update+=1;
        }, dresult);
        a(0) += dresult;
      });

      using policy_type = ReturnTeamPolicyType<execution_space>;
      Kokkos::parallel_for("TP", policy_type::get_policy(lvl_nodes,32,cuda1),
        KOKKOS_LAMBDA (const typename policy_type::PolicyType::member_type & member) {
        if (member.league_rank() == 0 && member.team_rank()==0)
          a(0) -= 3;

        member.team_barrier();

        double dresult = nlevels;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member,0,3), [=] (const int i, double &update) {
          update+=1;
        }, dresult);
        a(0) += dresult;
      });

        Kokkos::parallel_for("parfor_l_team_cudagraph", ReturnTeamPolicyType<execution_space>::get_policy(lvl_nodes,team_size,cuda1), LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count));

        node_count += hnodes_per_level(iter);
      }
    }
    cudaStreamEndCapture(stream1, &graph);

    // Create graphExec
    //cudaGraphExec_t graphExec;
    auto graphExec = lcl_cudagraph->cudagraphinstance;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Run graph
    Kokkos::fence();
    timer.reset();
    for(int r=0; r<R; r++) {
      cudaGraphLaunch(graphExec, stream1);
      int64_t s_result;
      if(std::is_same<result_space_type,Kokkos::CudaSpace>::value) {
        Kokkos::deep_copy(cuda1,s_result,result);
        cudaStreamSynchronize(stream1);
      } else {
        cudaStreamSynchronize(stream1);
        s_result = result();
      }
      printf("Result %i %li %li\n",r,s_result,int64_t(N)*int64_t(N-1)/2+int64_t(N)*2*(r+1));
    }
    double time2 = timer.seconds();
    printf("GraphTime<%s>: %e\n",result_space_type::name(),time2);
    Kokkos::fence();

    // Check whether its ok
    Kokkos::parallel_for("Check",N,KOKKOS_LAMBDA (const int i) {
      if(a(i)!=i+R*2) printf("Error: %i %i %i\n",i,a(i),i+R*2);
    });

  std::cout << "  Kokkos stream example check" << std::endl;
//  Kokkos::abort("Exit");
}


#else



#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

#ifdef TRISOLVE_TIMERS
  double time_outer = 0.0, time_inner_total = 0.0, time_setup = 0.0;
  Kokkos::Timer timer_total;
  Kokkos::Timer timer_inner;
  Kokkos::Timer timer_setup;
#endif

  auto nlevels = thandle.get_num_levels();
  // Keep this a host View, create device version and copy to back to host during scheduling
  // This requires making sure the host view in the handle is properly updated after the symbolic phase
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = thandle.get_host_nodes_per_level();
  //auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  //Kokkos::deep_copy(hnodes_per_level, nodes_per_level);  

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

  size_type node_count = 0;
#ifdef TRISOLVE_TIMERS
    // time for setup
    time_setup = timer_setup.seconds();
    timer_total.reset();
#endif

  // This must stay serial; would be nice to try out Cuda's graph stuff to reduce kernel launch overhead
#if defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION
  std::cout << "1 TESTING CUDAGRAPH" << std::endl;
  typename TriSolveHandle::SPTRSVcudaGraphWrapperType* lcl_cudagraph = thandle.get_sptrsvCudaGraph();
  auto stream = lcl_cudagraph->stream;
  Kokkos::Cuda kokkosstream1(stream);
  auto cudagraph = lcl_cudagraph->cudagraph;
  auto cudagraphinstance = lcl_cudagraph->cudagraphinstance;
  cudaStreamSynchronize(stream);

#else
  execution_space kokkosstream1();
#endif

  Kokkos::View<size_type*, Kokkos::HostSpace> hcum_npl("hcum_npl", nlevels);

  Kokkos::parallel_scan("row_map_rectspmtx scan", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, hcum_npl.extent(0)),
    KOKKOS_LAMBDA (const size_type i, size_type& update, const bool& final) {
      update += hnodes_per_level(i);
      if (final) {
        hcum_npl(i) = update;
      }
    });
  Kokkos::fence();

  std::cout << "hcum_npl(0) = " << hcum_npl(0) << "  hcum_npl(1) = " << hcum_npl(1) << std::endl;

        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        int team_size = thandle.get_team_size();
        team_size = team_size == -1 ? 64 : team_size;

  for ( size_type lvl = 0; lvl < nlevels; ++lvl ) {
#if defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION
  std::cout << "2 TESTING CUDAGRAPH" << std::endl;
  if (thandle.cudagraphCreated == false)
   {
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
#endif
    size_type lvl_nodes = hnodes_per_level(lvl);

#ifdef TRISOLVE_TIMERS
    timer_inner.reset();
#endif

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
      //if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 ) 
/*
        auto policy = ReturnTeamPolicyType<execution_space>::get_policy(lvl_nodes,team_size,kokkosstream1);

#ifdef TRILVLSCHED
        TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count);
#else
        LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#endif
          //Kokkos::parallel_for("parfor_l_team", policy_type( kokkosstream1, lvl_nodes , team_size ), tstf);
          Kokkos::parallel_for("parfor_l_team_cudagraph", policy, tstf);
*/
// Inlined...
          Kokkos::parallel_for("parfor_l_team_cudagraph", ReturnTeamPolicyType<execution_space>::get_policy(lvl_nodes,team_size,kokkosstream1), LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count));

/*
 // Test team policy - broken
        Kokkos::parallel_for("test", policy, KOKKOS_LAMBDA ( const typename policy_type::member_type &member ) {
          lhs(0) = 1;
        });

 // Test range policy - broken
        auto rpolicy = ReturnRangePolicyType<execution_space>::get_policy(0,4,kokkosstream1);
        Kokkos::parallel_for("test", rpolicy, KOKKOS_LAMBDA ( const int i ) {
          lhs(0) = 1;
        });
*/
      node_count += lvl_nodes;



#if defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION
  std::cout << "3 TESTING CUDAGRAPH" << std::endl;
   cudaStreamEndCapture(stream, &cudagraph );
  std::cout << "4 TESTING CUDAGRAPH" << std::endl;
   cudaGraphInstantiate(&cudagraphinstance, cudagraph, NULL, NULL, 0);
   thandle.cudagraphCreated = true;
#endif

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
#ifdef TRISOLVE_TIMERS
    // FIXME Adding Kokkos::fence() for timing purposes - is it necessary if running on a single stream???
   // Kokkos::fence();
    time_inner_total += timer_inner.seconds();
#endif
#if defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION
   } // scope for cudagraph if-block
#endif
#if defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION
  std::cout << "5 TESTING CUDAGRAPH" << std::endl;
   cudaGraphLaunch(cudagraphinstance, stream);
  std::cout << "6 TESTING CUDAGRAPH" << std::endl;
   cudaStreamSynchronize(stream);
//CU_STREAM_PER_THREAD
//CU_STREAM_LEGACY
#endif

  } // end for lvl
#ifdef TRISOLVE_TIMERS
  time_outer = timer_total.seconds();
  std::cout << "********************************" << std::endl; 
  std::cout << "  (l)tri_solve: setup = " << time_setup << std::endl;
  std::cout << "  (l)tri_solve: total loop = " << time_outer << std::endl;
  std::cout << "  (l)tri_solve: accum lvl solves = " << time_inner_total << std::endl;
  std::cout << "********************************" << std::endl; 
#endif

#endif
} // end lower_tri_solve_cg
#endif



template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void lower_tri_solve_ncg( TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs) {

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

#ifdef TRISOLVE_TIMERS
  double time_outer = 0.0, time_inner_total = 0.0, time_setup = 0.0;
  Kokkos::Timer timer_total;
  Kokkos::Timer timer_inner;
  Kokkos::Timer timer_setup;
#endif

  auto nlevels = thandle.get_num_levels();
  // Keep this a host View, create device version and copy to back to host during scheduling
  // This requires making sure the host view in the handle is properly updated after the symbolic phase
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = thandle.get_host_nodes_per_level();
  //auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  //Kokkos::deep_copy(hnodes_per_level, nodes_per_level);  

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

  size_type node_count = 0;
#ifdef TRISOLVE_TIMERS
    // time for setup
    time_setup = timer_setup.seconds();
    timer_total.reset();
#endif

  // This must stay serial; would be nice to try out Cuda's graph stuff to reduce kernel launch overhead

  typedef Kokkos::TeamPolicy<execution_space> policy_type;
  int team_size = thandle.get_team_size();
  team_size = team_size == -1 ? 64 : team_size;

  for ( size_type lvl = 0; lvl < nlevels; ++lvl ) 
  {
    size_type lvl_nodes = hnodes_per_level(lvl);

#ifdef TRISOLVE_TIMERS
    timer_inner.reset();
#endif

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
      //if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 ) 
      {

#ifdef TRILVLSCHED
        TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count);
#else
        LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#endif
          Kokkos::parallel_for("parfor_l_team", policy_type( lvl_nodes , team_size ), tstf);
      }

      node_count += lvl_nodes;

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
#ifdef TRISOLVE_TIMERS
    // FIXME Adding Kokkos::fence() for timing purposes - is it necessary if running on a single stream???
    Kokkos::fence();
    time_inner_total += timer_inner.seconds();
#endif
  } // scope for

#ifdef TRISOLVE_TIMERS
  time_outer = timer_total.seconds();
  std::cout << "********************************" << std::endl; 
  std::cout << "  (l)tri_solve: setup = " << time_setup << std::endl;
  std::cout << "  (l)tri_solve: total loop = " << time_outer << std::endl;
  std::cout << "  (l)tri_solve: accum lvl solves = " << time_inner_total << std::endl;
  std::cout << "********************************" << std::endl; 
#endif

} // end lower_tri_solve_ncg



template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void lower_tri_solve( TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs) {

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

#ifdef TRISOLVE_TIMERS
  double time_outer = 0.0, time_inner_total = 0.0, time_setup = 0.0;
  int tp1_ctr = 0;
  Kokkos::Timer timer_total;
  Kokkos::Timer timer_inner;
  Kokkos::Timer timer_setup;
#endif

  auto nlevels = thandle.get_num_levels();
  // Keep this a host View, create device version and copy to back to host during scheduling
  // This requires making sure the host view in the handle is properly updated after the symbolic phase
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = thandle.get_host_nodes_per_level();
  //auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  //Kokkos::deep_copy(hnodes_per_level, nodes_per_level);  

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

  size_type node_count = 0;
#ifdef TRISOLVE_TIMERS
    // time for setup
    time_setup = timer_setup.seconds();
    timer_total.reset();
#endif

  // This must stay serial; would be nice to try out Cuda's graph stuff to reduce kernel launch overhead


  for ( size_type lvl = 0; lvl < nlevels; ++lvl ) {
   {
    size_type lvl_nodes = hnodes_per_level(lvl);

#ifdef TRISOLVE_TIMERS
    timer_inner.reset();
#endif
    if ( lvl_nodes != 0 ) {

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
      if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP ) {
        Kokkos::parallel_for( "parfor_fixed_lvl", Kokkos::RangePolicy<execution_space>( node_count, node_count+lvl_nodes ), LowerTriLvlSchedRPSolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> (row_map, entries, values, lhs, rhs, nodes_grouped_by_level) );
      }
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 ) {
        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        int team_size = thandle.get_team_size();
    #ifdef TRISOLVE_TIMERS
        tp1_ctr++;
    #endif

#ifdef TRILVLSCHED
        TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count);
#else
        LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#endif
        if ( team_size == -1 )
          Kokkos::parallel_for("parfor_l_team", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
        else
          Kokkos::parallel_for("parfor_l_team", policy_type( lvl_nodes , team_size ), tstf);
      }
      // TP2 algorithm has issues with some offset-ordinal combo to be addressed
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
        typedef Kokkos::TeamPolicy<execution_space> tvt_policy_type;

        int team_size = thandle.get_team_size();
        if ( team_size == -1 ) {
          team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 64;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is responsible for processing team_size # nodes
        //       TeamThreadRange over number nodes of node_groups
        //       To avoid masking threads, 1 thread (team) per node in node_group (thread has full ownership of a node)
        //       ThreadVectorRange responsible for the actual solve computation
        //const int node_groups = team_size;
        const int node_groups = vector_size;

#ifdef TRILVLSCHED
        TriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count, vector_size, 0);
#else
        LowerTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count, node_groups);
#endif
        Kokkos::parallel_for("parfor_u_team_vector", tvt_policy_type( (int)std::ceil((float)lvl_nodes/(float)node_groups) , team_size, vector_size ), tstf);
      } // end elseif

      node_count += lvl_nodes;

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
    } // end if
#ifdef TRISOLVE_TIMERS
    // FIXME Adding Kokkos::fence() for timing purposes - is it necessary if running on a single stream???
    Kokkos::fence();
    time_inner_total += timer_inner.seconds();
#endif
   } // scope for if-block

  } // end for lvl
#ifdef TRISOLVE_TIMERS
  time_outer = timer_total.seconds();
  std::cout << "********************************" << std::endl; 
  std::cout << "  (l)tri_solve: setup = " << time_setup << std::endl;
  std::cout << "  (l)tri_solve: total loop = " << time_outer << std::endl;
  std::cout << "  (l)tri_solve: accum lvl solves = " << time_inner_total << std::endl;
  std::cout << "     solve calls = " << tp1_ctr << std::endl;
  std::cout << "********************************" << std::endl; 
#endif

} // end lower_tri_solve



template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void upper_tri_solve( TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs) {

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

#ifdef TRISOLVE_TIMERS
  double time_outer = 0.0, time_inner_total = 0.0, time_setup = 0.0;
  int tp1_ctr = 0;
  Kokkos::Timer timer_total;
  Kokkos::Timer timer_inner;
  Kokkos::Timer timer_setup;
#endif

  auto nlevels = thandle.get_num_levels();
  // Keep this a host View, create device version and copy to back to host during scheduling
  // This requires making sure the host view in the handle is properly updated after the symbolic phase
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = thandle.get_host_nodes_per_level();
  //auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  //Kokkos::deep_copy(hnodes_per_level, nodes_per_level);

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

  size_type node_count = 0;

#ifdef TRISOLVE_TIMERS
    // time for setup
    time_setup = timer_setup.seconds();
    timer_total.reset();
#endif

  // This must stay serial; would be nice to try out Cuda's graph stuff to reduce kernel launch overhead
  for ( size_type lvl = 0; lvl < nlevels; ++lvl ) {
    size_type lvl_nodes = hnodes_per_level(lvl);

#ifdef TRISOLVE_TIMERS
    timer_inner.reset();
#endif
    if ( lvl_nodes != 0 ) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif

      if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP ) {
        Kokkos::parallel_for( "parfor_fixed_lvl", Kokkos::RangePolicy<execution_space>( node_count, node_count+lvl_nodes ), UpperTriLvlSchedRPSolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> (row_map, entries, values, lhs, rhs, nodes_grouped_by_level) );
      }
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 ) {
        typedef Kokkos::TeamPolicy<execution_space> policy_type;

        int team_size = thandle.get_team_size();
    #ifdef TRISOLVE_TIMERS
        tp1_ctr++;
    #endif

#ifdef TRILVLSCHED
        TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, false, node_count);
#else
        UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#endif
        if ( team_size == -1 )
          Kokkos::parallel_for("parfor_u_team", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
        else
          Kokkos::parallel_for("parfor_u_team", policy_type( lvl_nodes , team_size ), tstf);
      }
      // TP2 algorithm has issues with some offset-ordinal combo to be addressed
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
        typedef Kokkos::TeamPolicy<execution_space> tvt_policy_type;

        int team_size = thandle.get_team_size();
        if ( team_size == -1 ) {
          team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 64;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is responsible for processing that many nodes
        //       TeamThreadRange over number nodes of node_groups
        //       To avoid masking threads, 1 thread (team) per node in node_group (thread has full ownership of a node)
        //       ThreadVectorRange responsible for the actual solve computation
        //const int node_groups = team_size;
        const int node_groups = vector_size;

#ifdef TRILVLSCHED
        TriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, false, node_count, vector_size, 0);
#else
        UpperTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count, node_groups);
#endif

        Kokkos::parallel_for("parfor_u_team_vector", tvt_policy_type( (int)std::ceil((float)lvl_nodes/(float)node_groups) , team_size, vector_size ), tstf);
      } // end elseif

      node_count += lvl_nodes;

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
    } // end if
#ifdef TRISOLVE_TIMERS
    // FIXME Adding Kokkos::fence() for timing purposes - is it necessary if running on a single stream???
    Kokkos::fence();
    time_inner_total += timer_inner.seconds();
#endif
  } // end for lvl
#ifdef TRISOLVE_TIMERS
  time_outer = timer_total.seconds();
  std::cout << "********************************" << std::endl; 
  std::cout << "  (u)tri_solve: setup = " << time_setup << std::endl;
  std::cout << "  (u)tri_solve: total loop = " << time_outer << std::endl;
  std::cout << "  (u)tri_solve: accum lvl solves = " << time_inner_total << std::endl;
  std::cout << "     solve calls = " << tp1_ctr << std::endl;
  std::cout << "********************************" << std::endl; 
#endif

} // end upper_tri_solve



// TODO is_lowertri unnecessary, just get it from the handle
template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void tri_solve_chain(TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs, const bool is_lowertri_) {

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

#ifdef TRISOLVE_TIMERS
  double time_outer = 0.0, time_full_solves = 0.0, time_chain_solves = 0.0, time_setup = 0.0, time_wrapped_ifelse = 0.0;

  int tp1_ctr = 0, chain_ctr = 0;

  double time_iter = 0.0;
  double time_extras = 0.0;

  Kokkos::Timer timer_extras;
  Kokkos::Timer timer_outer;
  Kokkos::Timer timer_full_solve;
  Kokkos::Timer timer_chain_solve;
  Kokkos::Timer timer_wrap_ifelse;
  Kokkos::Timer timer_setup;
#endif
  // Algorithm is checked before this function is called
  auto h_chain_ptr = thandle.get_host_chain_ptr();
  size_type num_chain_entries = thandle.get_num_chain_entries();

  // Keep this a host View, create device version and copy to back to host during scheduling
  // This requires making sure the host view in the handle is properly updated after the symbolic phase
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = thandle.get_host_nodes_per_level();

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

  const bool is_lowertri =  thandle.is_lower_tri();

  size_type node_count = 0;
#ifdef TRISOLVE_TIMERS
  // prep time
  time_setup = timer_setup.seconds(); 
  timer_outer.reset();
#endif

// REFACTORED to cleanup; next, need debug and timer routines
// Create a custom timer class
#ifdef SOLVE_IMPL_REFACTORING
  using policy_type = Kokkos::TeamPolicy<execution_space>;
  using large_cutoff_policy_type = Kokkos::TeamPolicy<LargerCutoffTag, execution_space>;
/*
  using TP1Functor = TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>;
  using LTP1Functor = LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>;
  using UTP1Functor = UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>;
  using LSingleBlockFunctor = LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>;
  using USingleBlockFunctor = UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>;
*/
  using SingleBlockFunctor = TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType>;

  int team_size = thandle.get_team_size();

  auto cutoff = thandle.get_chain_threshold();
  int team_size_singleblock = team_size;

  // FIXME - should respect user's team_size
  // FIXME - move this decision inside???
  // Enumerate options
  // ts -1,0 | cu 0 - select default ts == 1
  // ts -1,0 | cu > 0 - select default ts; restriction: ts <= tsmax (auto)
  // ts > 0 | cu 0 - set
  // ts > 0 | cu > 0 - set
  // Controls ts,cu > 0
  // co > ts  - not all rows can be mapped to a thread - must call largercutoff impl
  // co <= ts - okay, kernel must be careful not to access out-of-bounds; some threads idol
  if (team_size_singleblock <= 0 && cutoff == 0) {
    team_size_singleblock = 1;
    // If cutoff == 0, no single-block calls will be made, team_size_singleblock is unimportant
  }
  /*
  else if (team_size_singleblock <= 0 && cutoff > 0) {
    // Need to select team_size_sb based on functor, but it depends at runtime on different values...
    team_size_singleblock = cutoff;
  } // FIXME - this will break if cutoff exceeds max team size of hardware
  */


  // This is only necessary for Lower,UpperTri functor versions; else, is_lowertri can be passed as arg to the generic Tri functor...
  if (is_lowertri) {

    for ( size_type chainlink = 0; chainlink < num_chain_entries; ++chainlink ) {
      size_type schain = h_chain_ptr(chainlink);
      size_type echain = h_chain_ptr(chainlink+1);

      if ( echain - schain == 1 ) {

        // if team_size is -1 (unset), get recommended size from Kokkos
#ifdef TRILVLSCHED
        TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count);
#else
        LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#endif
        if (team_size == - 1) {
          team_size = policy_type(1, 1, 1).team_size_recommended(tstf, Kokkos::ParallelForTag());
        }

        // TODO To use cudagraph here, need to know how many non-unit chains there are, create a graph for each and launch accordingly
        size_type lvl_nodes = hnodes_per_level(schain); //lvl == echain????
        Kokkos::parallel_for("parfor_l_team_chain1", policy_type( lvl_nodes , team_size ), tstf);
        node_count += lvl_nodes;
        //std::cout << "  schain: " << schain << "  lvl_nodes: " << lvl_nodes << "  updated node_count: " << node_count << std::endl;

      }
      else {
        size_type lvl_nodes = 0;

        for (size_type i = schain; i < echain; ++i) {
          lvl_nodes += hnodes_per_level(i);
        }

        if (team_size_singleblock <= 0) {
          team_size_singleblock = policy_type(1, 1, 1).team_size_recommended(SingleBlockFunctor(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, is_lowertri), Kokkos::ParallelForTag());
        }

        if (cutoff <= team_size_singleblock) {
#ifdef TRILVLSCHED
          TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true);
#else
          LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
#endif
          Kokkos::parallel_for("parfor_l_team_chainmulti", policy_type(1, team_size_singleblock), tstf);
        }
        else {
          // team_size_singleblock < cutoff => kernel must allow for a block-stride internally
#ifdef TRILVLSCHED
          TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true, 0, cutoff);
#else
          LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, cutoff);
#endif
          Kokkos::parallel_for("parfor_l_team_chainmulti_cutoff", large_cutoff_policy_type(1, team_size_singleblock), tstf);
        }
        node_count += lvl_nodes;
      }
      Kokkos::fence(); // TODO - is this necessary? that is, can the parallel_for launch before the s/echain values have been updated?
    }

  }
  else {

    for ( size_type chainlink = 0; chainlink < num_chain_entries; ++chainlink ) {
      size_type schain = h_chain_ptr(chainlink);
      size_type echain = h_chain_ptr(chainlink+1);

      if ( echain - schain == 1 ) {

        // if team_size is -1 (unset), get recommended size from Kokkos
#ifdef TRILVLSCHED
        TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, is_lowertri, node_count);
#else
        UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#endif
        if (team_size == - 1) {
          team_size = policy_type(1, 1, 1).team_size_recommended(tstf, Kokkos::ParallelForTag());
        }

        // TODO To use cudagraph here, need to know how many non-unit chains there are, create a graph for each and launch accordingly
        size_type lvl_nodes = hnodes_per_level(schain); //lvl == echain????
        Kokkos::parallel_for("parfor_u_team_chain1", policy_type( lvl_nodes , team_size ), tstf);
        node_count += lvl_nodes;
        //std::cout << "  schain: " << schain << "  lvl_nodes: " << lvl_nodes << "  updated node_count: " << node_count << std::endl;

      }
      else {
        size_type lvl_nodes = 0;

        for (size_type i = schain; i < echain; ++i) {
          lvl_nodes += hnodes_per_level(i);
        }

        if (team_size_singleblock <= 0) {
          //team_size_singleblock = policy_type(1, 1, 1).team_size_recommended(SingleBlockFunctor(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, is_lowertri, node_count), Kokkos::ParallelForTag());
          team_size_singleblock = policy_type(1, 1, 1).team_size_recommended(SingleBlockFunctor(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, is_lowertri), Kokkos::ParallelForTag());
        }

        if (cutoff <= team_size_singleblock) {
#ifdef TRILVLSCHED
          TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, is_lowertri);
#else
          UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
#endif
          Kokkos::parallel_for("parfor_u_team_chainmulti", policy_type(1, team_size_singleblock), tstf);
        }
        else {
          // team_size_singleblock < cutoff => kernel must allow for a block-stride internally
#ifdef TRILVLSCHED
          TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, is_lowertri, 0, cutoff);
#else
          UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, cutoff);
#endif
          Kokkos::parallel_for("parfor_u_team_chainmulti_cutoff", large_cutoff_policy_type(1, team_size_singleblock), tstf);
        }
        node_count += lvl_nodes;
      }
      Kokkos::fence(); // TODO - is this necessary? that is, can the parallel_for launch before the s/echain values have been updated?
    }

  }
#else
  for ( size_type chainlink = 0; chainlink < num_chain_entries; ++chainlink ) {
  #ifdef TRISOLVE_TIMERS
      timer_extras.reset();
  #endif
    size_type schain = h_chain_ptr(chainlink);
    size_type echain = h_chain_ptr(chainlink+1);
  #ifdef TRISOLVE_TIMERS
      time_extras += timer_extras.seconds();
  #endif

  #ifdef TRISOLVE_TIMERS
     // fenced solve time
    timer_wrap_ifelse.reset();
  #endif
    if ( echain - schain == 1 ) {
      //std::cout << "Call regular single-link TP - chainlink: " << chainlink << std::endl;
      // run normal algm as this is a single level
      // schain should.... map to the level....
  #ifdef TRISOLVE_TIMERS
      timer_full_solve.reset();
      // full-solve time
      tp1_ctr++;
      //std::cout << "  *** Calling non-single-block solve *** " << std::endl;
      //std::cout << "      team_size = " << team_size << std::endl;
      //std::cout << "      lvl_nodes = " << lvl_nodes << std::endl;
  #endif
        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        int team_size = thandle.get_team_size();

        size_type lvl_nodes = hnodes_per_level(schain); //lvl == echain????

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
        if (is_lowertri) {
          // TODO Time changes between merged functor and individuals
#ifdef TRILVLSCHED
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count);
#else
          LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#endif
          if ( team_size == -1 )
            Kokkos::parallel_for("parfor_l_team_chain1auto", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
          else
            Kokkos::parallel_for("parfor_l_team_chain1", policy_type( lvl_nodes , team_size ), tstf);
        }
        else {
#ifdef TRILVLSCHED
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, false, node_count);
#else
          UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#endif
          if ( team_size == -1 )
            Kokkos::parallel_for("parfor_u_team_chain1auto", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
          else
            Kokkos::parallel_for("parfor_u_team_chain1", policy_type( lvl_nodes , team_size ), tstf);
        }
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif

        // echain is offset into level... ??
        node_count += lvl_nodes;
        //std::cout << "  schain: " << schain << "  lvl_nodes: " << lvl_nodes << "  updated node_count: " << node_count << std::endl;

      // TODO Test this inside if-else vs here
      Kokkos::fence();
  #ifdef TRISOLVE_TIMERS
      // full-solve time
      time_iter = timer_full_solve.seconds();
      time_full_solves += time_iter;
      std::cout << "  tp1 iter: " << tp1_ctr << "  time_iter = " << time_iter << std::endl;
      //time_full_solves += timer_full_solve.seconds();
  #endif
    }
    else {
      //std::cout << "Call multi-link single-block TP - chainlink: " << chainlink << std::endl;
      // run single_block algm, pass echain and schain as args
  #ifdef TRISOLVE_TIMERS
      timer_chain_solve.reset();
     // full-solve time
      chain_ctr++;
      //std::cout << "  *** Calling single-block solve *** " << std::endl;
      //std::cout << "      team_size = " << team_size << "  cutoff = " << cutoff << std::endl;
      //std::cout << "      lvl_nodes = " << lvl_nodes << std::endl;
  #endif
        size_type lvl_nodes = 0;

        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        typedef Kokkos::TeamPolicy<LargerCutoffTag, execution_space> large_cutoff_policy_type;
        auto cutoff = thandle.get_chain_threshold();
        // FIXME - should respect user's team_size
        //const int team_size = cutoff;
        auto team_size = thandle.get_team_size();
        if (team_size <= 0 && cutoff == 0) { team_size = 1; }
        else if (team_size <= 0 && cutoff > 0) { team_size = cutoff; }
  #ifdef TRISOLVE_TIMERS
     // full-solve time
      chain_ctr++;
      std::cout << "  *** Calling single-block solve *** " << std::endl;
      std::cout << "      team_size = " << team_size << "  cutoff = " << cutoff << std::endl;
      std::cout << "      lvl_nodes = " << lvl_nodes << std::endl;
      timer_chain_solve.reset();
  #endif
//        const int team_size = std::is_same<typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>::value ? 1 : 256; // TODO chainlink cutoff hard-coded to 256: make this a "threshold" parameter in the handle

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
        // FIXME should be able to simply pass is_lowertri to functor, no need for "if is_lowertri else" branches when using TriLvl* functors
        // Will the cost in perf making a runtime if-check during each solve though??
        if (is_lowertri) {
          if (cutoff <= team_size) {
#ifdef TRILVLSCHED
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true);
#else
            LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
#endif
            Kokkos::parallel_for("parfor_l_team_chainmulti", policy_type( 1, team_size ), tstf);
          }
          else {
            // team_size < cutoff => kernel must allow for a block-stride internally
#ifdef TRILVLSCHED
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true, 0, cutoff);
#else
            LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, cutoff);
#endif
            Kokkos::parallel_for("parfor_l_team_chainmulti_cutoff", large_cutoff_policy_type( 1, team_size ), tstf);
          }
        }
        else {
          if (cutoff <= team_size) {
#ifdef TRILVLSCHED
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, false);
#else
            UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
#endif
            Kokkos::parallel_for("parfor_u_team_chainmulti", policy_type( 1, team_size ), tstf);
          }
          else {
            // team_size < cutoff => kernel must allow for a block-stride internally
#ifdef TRILVLSCHED
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, false, 0, cutoff);
#else
            UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, cutoff);
#endif
            Kokkos::parallel_for("parfor_u_team_chainmulti_cutoff", large_cutoff_policy_type( 1, team_size ), tstf);
          }
        }

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
        for (size_type i = schain; i < echain; ++i) {
          lvl_nodes += hnodes_per_level(i);
        }
        node_count += lvl_nodes;
        //std::cout << "  echain: " << echain << "  lvl_nodes: " << lvl_nodes << "  updated node_count: " << node_count << std::endl;

      // TODO Test this inside if-else vs here
      Kokkos::fence();
  #ifdef TRISOLVE_TIMERS
     // full-solve time
      time_iter = timer_chain_solve.seconds();
      time_chain_solves += time_iter;
      std::cout << "  chain iter: " << chain_ctr << "  time_iter = " << time_iter << std::endl;
      //time_chain_solves += timer_chain_solve.seconds();
  #endif
    } // end else

    // TODO Test this inside if-else vs here
    //Kokkos::fence();
  #ifdef TRISOLVE_TIMERS
     // fenced solve time
     time_wrapped_ifelse += timer_wrap_ifelse.seconds();
  #endif
  } // end for chainlink
#endif
#ifdef TRISOLVE_TIMERS
   // Total chain time
   time_outer = timer_outer.seconds(); 

  std::cout << "********************************" << std::endl; 
  std::cout << "  tri_solve_chain: setup = " << time_setup << std::endl;
  std::cout << "  tri_solve_chain: total loop = " << time_outer << std::endl;
  std::cout << "  tri_solve_chain: full lvl solves = " << time_full_solves << std::endl;
  std::cout << "      solve count = " << tp1_ctr << std::endl;
  std::cout << "  tri_solve_chain: chain solves = " << time_chain_solves << std::endl;
  std::cout << "      single-block solve count = " << chain_ctr << std::endl;
  std::cout << "  tri_solve_chain: total if-else solve times = " << time_wrapped_ifelse << std::endl;
  std::cout << "  tri_solve_chain: extras = " << time_extras << std::endl;
  std::cout << "********************************" << std::endl; 
#endif

} // end tri_solve_chain



#ifdef DENSEPARTITION
template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void tri_solve_partition_dense(TriSolveHandle & thandle, const RowMapType frow_map, const EntriesType entries, const ValuesType values, const RHSType & frhs, LHSType & flhs, const bool is_lowertri) {

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

// Part 1. Sparse partition of the matrix, computation done as in other algorithms, just need to take subviews of the input view arrays
//  auto dense_row_start = thandle.get_dense_partition_row_start();
//  auto dense_partition_nrows = thandle.get_dense_partition_nrows() ;
#ifdef TRISOLVE_TIMERS
  double time_outer = 0.0, time_full_solves = 0.0, time_chain_solves = 0.0, time_setup = 0.0, time_wrapped_ifelse = 0.0;

  int tp1_ctr = 0, chain_ctr = 0;

  double time_iter = 0.0;

  Kokkos::Timer timer_outer;
  Kokkos::Timer timer_full_solve;
  Kokkos::Timer timer_chain_solve;
  Kokkos::Timer timer_wrap_ifelse;
  Kokkos::Timer timer_setup;
#endif

  auto persist_sptrimtx_row_start = thandle.get_persist_sptrimtx_row_start();
  auto persist_sptrimtx_nrows = thandle.get_persist_sptrimtx_nrows();
  auto persist_sptrimtx_row_end = persist_sptrimtx_row_start + persist_sptrimtx_nrows;


  // "Shifted" row_map and vectors; we will still use the original entries and values, as shifted row_map will still index into them,
  // but need to offset the colid as well for upper_tri solves
  auto row_map = Kokkos::subview(frow_map, Kokkos::pair<size_type,size_type>(persist_sptrimtx_row_start, persist_sptrimtx_row_end+1));
  auto rhs = Kokkos::subview(frhs, Kokkos::pair<size_type,size_type>(persist_sptrimtx_row_start, persist_sptrimtx_row_end));
  auto lhs = Kokkos::subview(flhs, Kokkos::pair<size_type,size_type>(persist_sptrimtx_row_start, persist_sptrimtx_row_end));


#ifdef PRINT1DVIEWS
  print_view1d_solve(frow_map);
  print_view1d_solve(entries);
  print_view1d_solve(values);
  print_view1d_solve(frhs);
  print_view1d_solve(flhs);

  print_view1d_solve(row_map);
  print_view1d_solve(rhs);
  print_view1d_solve(lhs);
#endif


#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
  // Algorithm is checked before this function is called
  auto h_chain_ptr = thandle.get_host_chain_ptr();
  size_type num_chain_entries = thandle.get_num_chain_entries();

  // Keep this a host View, create device version and copy to back to host during scheduling
  // This requires making sure the host view in the handle is properly updated after the symbolic phase
  auto nodes_per_level = thandle.get_nodes_per_level();
  auto hnodes_per_level = thandle.get_host_nodes_per_level();
  //auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  //Kokkos::deep_copy(hnodes_per_level, nodes_per_level);

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

#ifdef USEDIAGVALUES
  auto fdiagonal_values = thandle.get_diagonal_values();
  auto diagonal_values = Kokkos::subview(fdiagonal_values, Kokkos::pair<size_type,size_type>(persist_sptrimtx_row_start, persist_sptrimtx_row_end));
#endif

  size_type node_count = 0;
#ifdef TRISOLVE_TIMERS
  // prep time
  time_setup = timer_setup.seconds(); 
  timer_outer.reset();
#endif


#ifdef DENSEPARTITION
  size_type dense_nrows = thandle.get_dense_partition_nrows();
#else
  size_type dense_nrows = 0;
#endif

#ifdef SOLVE_DEBUG_OUTPUT
  std::cout << "  solve: dense_nrows = " << dense_nrows << std::endl;
  std::cout << "  solve: num_chain_entries = " << num_chain_entries << std::endl;
      std::cout << "    h_chain_ptr.extent(0) = " << h_chain_ptr.extent(0) << std::endl;
      std::cout << "    hnodes_per_level.extent(0) = " << hnodes_per_level.extent(0) << std::endl;
#endif
  
  
  for ( size_type chainlink = 0; chainlink < num_chain_entries; ++chainlink ) {
    size_type schain = h_chain_ptr(chainlink);
    size_type echain = h_chain_ptr(chainlink+1);

  #ifdef TRISOLVE_TIMERS
     // fenced solve time
    timer_wrap_ifelse.reset();
  #endif
    if ( echain - schain == 1 ) {
#ifdef LVL_OUTPUT_INFO
      std::cout << "Call regular single-link TP - chainlink: " << chainlink << std::endl;
#endif
      // run normal algm as this is a single level
      // schain should.... map to the level....
      typedef Kokkos::TeamPolicy<execution_space> policy_type;
      int team_size = thandle.get_team_size();

      size_type lvl_nodes = hnodes_per_level(schain); //lvl == echain????
#ifdef SOLVE_DEBUG_OUTPUT
      std::cout << "  *** Calling non-single-block solve *** " << std::endl;
      std::cout << "      team_size = " << team_size << std::endl;
      std::cout << "      lvl_nodes = " << lvl_nodes << std::endl;
#endif
  #ifdef TRISOLVE_TIMERS
      // full-solve time
      tp1_ctr++;
      timer_full_solve.reset();
  #endif

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
        // FIXME should be able to simply pass is_lowertri to functor, no need for "if is_lowertri else" branches when using TriLvl* functors
        // Will the cost in perf making a runtime if-check during each solve though??
        if (is_lowertri) {
          // TODO Time changes between merged functor and individuals
         if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1)
         {
          //LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#ifdef USEDIAGVALUES
          TriLvlSchedTP1SolverFunctorDiagValues<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, diagonal_values, true, node_count, dense_nrows);
#else
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count, dense_nrows);
#endif

          if ( team_size <= 0 )
            Kokkos::parallel_for("parfor_l_team_chain1autodense", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
          else
            Kokkos::parallel_for("parfor_l_team_chain1dense", policy_type( lvl_nodes , team_size ), tstf);
         }
         else if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP2)
         {
           if ( team_size <= 0 )
           {
             team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 64;
           }
           int vector_size = thandle.get_team_size();
           if ( vector_size == -1 ) {
             vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
           }

           TriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count, vector_size, dense_nrows);

           const int num_teams = (int)std::ceil((float)lvl_nodes/(float)vector_size);
           Kokkos::parallel_for("parfor_l_team_chaindense_tp2", policy_type( num_teams, team_size , vector_size ), tstf);
         }
        }
        else {
         if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1)
         {
          //UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
#ifdef USEDIAGVALUES
          TriLvlSchedTP1SolverFunctorDiagValues<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, diagonal_values, false, node_count, dense_nrows);
#else
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, false, node_count, dense_nrows);
#endif

          if ( team_size <= 0 )
            Kokkos::parallel_for("parfor_u_team_chain1autodense", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
          else
            Kokkos::parallel_for("parfor_u_team_chain1dense", policy_type( lvl_nodes , team_size ), tstf);
         }
         else if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP2)
         {
           if ( team_size <= 0 )
           {
             team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 64;
           }
           int vector_size = thandle.get_team_size();
           if ( vector_size == -1 ) {
             vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
           }

           TriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, false, node_count, vector_size, dense_nrows);

           const int num_teams = (int)std::ceil((float)lvl_nodes/(float)vector_size);
           Kokkos::parallel_for("parfor_u_team_chaindense_tp2", policy_type( num_teams, team_size , vector_size ), tstf);
         }
        }
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif

        // echain is offset into level... ??
        node_count += lvl_nodes;
        //std::cout << "  lvl_nodes = " << lvl_nodes << std::endl;
        //std::cout << "  schain: " << schain << "  lvl_nodes: " << lvl_nodes << "  updated node_count: " << node_count << std::endl;

      // TODO Test this inside if-else vs here
      Kokkos::fence();
  #ifdef TRISOLVE_TIMERS
      // full-solve time
      time_iter = timer_full_solve.seconds();
      time_full_solves += time_iter;
      //std::cout << "  tp1 iter: " << tp1_ctr << "  time_iter = " << time_iter << std::endl;
      //time_full_solves += timer_full_solve.seconds();
  #endif
    }
    else {
#ifdef LVL_OUTPUT_INFO
      std::cout << "Call multi-link single-block TP - chainlink: " << chainlink << std::endl;
#endif
      // run single_block algm, pass echain and schain as args
        size_type lvl_nodes = 0;

        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        typedef Kokkos::TeamPolicy<LargerCutoffTag, execution_space> large_cutoff_policy_type;
        auto cutoff = thandle.get_chain_threshold();
        // FIXME Ignore user-specified team_size????
        //const int team_size = cutoff == 0 ? 1 : cutoff;
        // team_size < cutoff not supported for this case...
        auto team_size = thandle.get_team_size();
        if (team_size <= 0 && cutoff == 0) { team_size = 1; }
        else if (team_size <= 0 && cutoff > 0) { team_size = cutoff; }
#ifdef SOLVE_DEBUG_OUTPUT
      std::cout << "  *** Calling single-block solve *** " << std::endl;
      std::cout << "      team_size = " << team_size << "  cutoff = " << cutoff << std::endl;
      std::cout << "      lvl_nodes = " << lvl_nodes << std::endl;
#endif
  #ifdef TRISOLVE_TIMERS
     // full-solve time
      chain_ctr++;
      timer_chain_solve.reset();
  #endif
//        const int team_size = std::is_same<typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>::value ? 1 : 256; // TODO chainlink cutoff hard-coded to 256: make this a "threshold" parameter in the handle

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
        for (size_type i = schain; i < echain; ++i) {
          lvl_nodes += hnodes_per_level(i);
        }

        // FIXME should be able to simply pass is_lowertri to functor, no need for "if is_lowertri else" branches when using TriLvl* functors
        // Will the cost in perf making a runtime if-check during each solve though??
        if (is_lowertri) {
          if (cutoff <= team_size) {
            std::cout << "cutoff <= team_size" << std::endl;
//          LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);

           //if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1)
           {
#ifdef USEDIAGVALUES
            TriLvlSchedTP1SingleBlockFunctorDiagValues<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, diagonal_values, node_count, schain, echain, true, dense_nrows);
#else
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true, dense_nrows);
#endif
            Kokkos::parallel_for("parfor_l_team_chainmulti", policy_type( 1, team_size ), tstf);
           }
          }
          else {
            std::cout << "cutoff > team_size" << std::endl;
            // team_size < cutoff => kernel must allow for a block-stride internally
//          LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);

           //if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1)
           {
#ifdef USEDIAGVALUES
            TriLvlSchedTP1SingleBlockFunctorDiagValues<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, diagonal_values, node_count, schain, echain, true, dense_nrows, cutoff);
#else
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true, dense_nrows, cutoff);
#endif
            Kokkos::parallel_for("parfor_l_team_chainmulti_cutoff", large_cutoff_policy_type( 1, team_size ), tstf);
           }
          }
        }
        else {
          if (cutoff <= team_size) {
//          UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);

           //if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1)
           {
#ifdef USEDIAGVALUES
            TriLvlSchedTP1SingleBlockFunctorDiagValues<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, diagonal_values, node_count, schain, echain, false, dense_nrows);
#else
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, false, dense_nrows);
#endif
            Kokkos::parallel_for("parfor_u_team_chainmulti", policy_type( 1, team_size ), tstf);
           }
          }
          else {
            // team_size < cutoff => kernel must allow for a block-stride internally
//          UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);

           //if (thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1)
           {
#ifdef USEDIAGVALUES
            TriLvlSchedTP1SingleBlockFunctorDiagValues<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, diagonal_values, node_count, schain, echain, false, dense_nrows, cutoff);
#else
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, false, dense_nrows, cutoff);
#endif
            Kokkos::parallel_for("parfor_u_team_chainmulti_cutoff", large_cutoff_policy_type( 1, team_size ), tstf);
           }
          }
        }

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif

        //std::cout << "  lvl_nodes = " << lvl_nodes << std::endl;
      node_count += lvl_nodes;
        //std::cout << "  echain: " << echain << "  lvl_nodes: " << lvl_nodes << "  updated node_count: " << node_count << std::endl;

      // TODO Test this inside if-else vs here
      Kokkos::fence();
  #ifdef TRISOLVE_TIMERS
     // full-solve time
      time_iter = timer_chain_solve.seconds();
      time_chain_solves += time_iter;
      //std::cout << "  chain iter: " << chain_ctr << "  time_iter = " << time_iter << std::endl;
      //time_chain_solves += timer_chain_solve.seconds();
  #endif
    } // end else

    // TODO Test this inside if-else vs here
    //Kokkos::fence();
  #ifdef TRISOLVE_TIMERS
     // fenced solve time
     auto tmptime = timer_wrap_ifelse.seconds();
     time_wrapped_ifelse += tmptime;
  #ifdef TRISOLVE_TIMERS_ITER_OUTPUT
     std::cout << "    time iter: " << tp1_ctr << "  chain_ctr: " << chain_ctr << "  time: " << tmptime << std::endl;
  #endif
     //time_wrapped_ifelse += timer_wrap_ifelse.seconds();
  #endif
  } // end for chainlink

#ifdef PRINT1DVIEWS
  std::cout << "Complete sptrimtx solution" << std::endl;
  print_view1d_solve(lhs);
#endif

// Part 2. gemv, set xp <- bp - Mp*xknown
//                 lhsp <- rhsp - Mp*lhs  lhs the subview from part 1.
// Process:
//           1. lhsp = Kokkos::subview(flhs, pair(cutoff,nrows); rhsp = Kokkos::subview(frhs, pair(cutoff,nrows); deep_copy(lhsp, rhsp);
//           2. gemv("N", -1.0, dense_mtx, lhs, 1.0, lhsp); (where rhsp i.e. b was copied into lhsp, and lhs is the solution from part 1)
//           3. Kokkos::fence(); ?
  //auto dense_mtx = thandle.get_dense_mtx_partition(); // FIXME Need to remove and replace with subview components in sparse components

  //Create KokkosSparse::CrsMatrix from modified row_map, entries, and values
  // This may require "shifting" the modified row_map, subview the entries and values and shift the entries array by subtracting out colid shift
  // lowertri
 if (dense_nrows > 0) {
  #ifdef TRISOLVE_TIMERS
  double time_rectspmtx_total = 0.0, time_spmv = 0.0;
  Kokkos::Timer timer_rectspmtx;
  #endif

  auto row_map_rectspmtx = thandle.get_row_map_rectspmtx();
  auto entries_rectspmtx = thandle.get_entries_rectspmtx();
  auto values_rectspmtx = thandle.get_values_rectspmtx();

  typedef typename EntriesType::value_type crs_lno_t;
  typedef typename ValuesType::value_type crs_scalar_t;
  typedef typename RowMapType::value_type  crs_size_type;
  typedef typename RowMapType::execution_space crs_exec_space;

  crs_lno_t rectspmtx_nrows = thandle.get_rectspmtx_nrows();
  crs_lno_t rectspmtx_ncols = thandle.get_rectspmtx_ncols();
  crs_size_type rectspmtx_nnz = thandle.get_nnz_rectspmtx();

  auto rectspmtx_row_start = thandle.get_rectspmtx_row_start();

  auto lhsp = Kokkos::subview(flhs, Kokkos::pair<size_type, size_type>(rectspmtx_row_start, rectspmtx_row_start + rectspmtx_nrows)); 
  auto rhsp = Kokkos::subview(frhs, Kokkos::pair<size_type, size_type>(rectspmtx_row_start, rectspmtx_row_start + rectspmtx_nrows)); 
  Kokkos::deep_copy(lhsp, rhsp);

#ifdef PRINT1DVIEWS
  std::cout << "Pre spmv rhsp" << std::endl;
  print_view1d_solve(rhsp);
#endif

  KokkosSparse::CrsMatrix<crs_scalar_t, crs_lno_t, crs_exec_space, void, crs_size_type> crs_rectspmtx("rect_spmtx", rectspmtx_nrows, rectspmtx_ncols, rectspmtx_nnz, values_rectspmtx, row_map_rectspmtx, entries_rectspmtx);  

  #ifdef TRISOLVE_TIMERS
  time_rectspmtx_total += timer_rectspmtx.seconds();
  timer_rectspmtx.reset();
  #endif
  // lhsp <- 1.0*lhsp + -1.0*crs_rectspmtx*lhs
  // y <- beta*y + alpha*A*x
  // spmv("trans", alpha, A, x, beta, y)
  KokkosSparse::spmv("N", -1.0, crs_rectspmtx, lhs, 1.0, lhsp); //(where rhsp i.e. b was copied into lhsp, and lhs is the solution from part 1)

  // TODO Is this necessary???
  Kokkos::fence();
  #ifdef TRISOLVE_TIMERS
  time_spmv += timer_rectspmtx.seconds();
  #endif

#ifdef PRINT1DVIEWS
  std::cout << "Post-spmv intermediate lhsp" << std::endl;
  print_view1d_solve(lhsp);
#endif


// Part 3. dense trisolve for remaining dense portion of x - use x as rhs and lhs in this step
//           * Treat lhsp as partially updated rhsp, overwrite for final result

  auto dense_trimtx = thandle.get_dense_trimtx();

// cublass API to dense trsv
//cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
//                           cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)

  #ifdef TRISOLVE_TIMERS
  double time_densetri = 0.0;
  Kokkos::Timer timer_densetri;
  #endif
  // beginnings of the call...
  // Need guards for Cuda being enabled AND active, etc
#if defined(KOKKOS_ENABLED_CUDA)

  cublasStatus_t stat;
  cublasHandle_t cublashandle ;

  stat = cublasCreate(&cublashandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS handle created failed\n");
        return EXIT_FAILURE;
    }
  cublasFillMode_t uplo = is_lowertri ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
  //cublasDiagType_t diag = CUBLAS_DIAG_UNIT;
  bool tri_is_lr = std::is_same<Kokkos::LayoutRight, typename TriSolveHandle::mtx_scalar_view_t::array_layout >::value;
  const int AST = tri_is_lr?dense_trimtx.stride(0):dense_trimtx.stride(1);
  LDA = AST == 0 ? 1 : AST;

  stat = cublasDtrsv(cublashandle, uplo, trans, diag, dense_trimtx.extent(0), dense_trimtx.data(), LDA, lhsp.data(), 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS Dtrsv created failed\n");
        return EXIT_FAILURE;
    }

  stat = cublasDestroy(cublashandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS handle destroy failed\n");
        return EXIT_FAILURE;
    }
#else
    // Call BLAS routine...
  Kokkos::parallel_for("Call batched_trsv", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), 
    KOKKOS_LAMBDA( const int i ) {
        if (is_lowertri) {
        KokkosBatched::SerialTrsv<
          KokkosBatched::Uplo::Lower,
          KokkosBatched::Trans::NoTranspose,
          KokkosBatched::Diag::NonUnit,
          KokkosBatched::Algo::Trsv::Unblocked
        >::invoke(1.0, dense_trimtx, lhsp);
      }
      else {
        KokkosBatched::SerialTrsv<
          KokkosBatched::Uplo::Upper,
          KokkosBatched::Trans::NoTranspose,
          KokkosBatched::Diag::NonUnit,
          KokkosBatched::Algo::Trsv::Unblocked
        >::invoke(1.0, dense_trimtx, lhsp);
      }
    });
#endif

  Kokkos::fence();
  std::cout << "solve complete" << std::endl;

  #ifdef TRISOLVE_TIMERS
  time_densetri += timer_densetri.seconds();
  std::cout << "********************************" << std::endl; 
  std::cout << "  tri_solve_partition: spmv setup = " << time_rectspmtx_total << std::endl;
  std::cout << "  tri_solve_partition: spmv time = " << time_spmv << std::endl;
  std::cout << "  tri_solve_partition: dense_tri time = " << time_densetri << std::endl;
  std::cout << "********************************" << std::endl; 
  #endif

#ifdef PRINT1DVIEWS
  std::cout << "Output dense partition solution" << std::endl;
  print_view1d_solve(lhsp);
#endif
 }

#ifdef TRISOLVE_TIMERS
  // Total chain time
  time_outer = timer_outer.seconds(); 

  std::cout << "********************************" << std::endl; 
  std::cout << "  tri_solve_chain: setup = " << time_setup << std::endl;
  std::cout << "  tri_solve_chain: total loop = " << time_outer << std::endl;
  std::cout << "  tri_solve_chain: full lvl solves = " << time_full_solves << std::endl;
  std::cout << "      solve count = " << tp1_ctr << std::endl;
  std::cout << "  tri_solve_chain: chain solves = " << time_chain_solves << std::endl;
  std::cout << "      single-block solve count = " << chain_ctr << std::endl;
  std::cout << "  tri_solve_chain: total if-else solve times = " << time_wrapped_ifelse << std::endl;

  std::cout << "********************************" << std::endl; 
#endif

} // end tri_solve_partition_dense
#endif


template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void solve_impl(TriSolveHandle & thandle, const RowMapType &row_map, const EntriesType &entries, const ValuesType &values, const RHSType & rhs, LHSType &lhs) {

    const bool is_lower = thandle.is_lower_tri();

      if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1 || thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
        /*
        if ( thandle->is_symbolic_complete() == false ) {
          if (is_lower)
          KokkosSparse::Impl::Experimental::lower_tri_symbolic(thandle, row_map, entries);
          else
          KokkosSparse::Impl::Experimental::upper_tri_symbolic(thandle, row_map, entries);
        }
        */
        if (is_lower) {
#if defined(LTCUDAGRAPHTEST) && defined(KOKKOS_ENABLE_CUDA) && 10000 < CUDA_VERSION
          std::cout << "  lower_tri_solve_cg version" << std::endl;
          KokkosSparse::Impl::Experimental::lower_tri_solve_cg(thandle, row_map, entries, values, rhs, lhs);
        //KokkosSparse::Impl::Experimental::lower_tri_solve_ncg( *thandle, row_map, entries, values, rhs, lhs);
#else
          std::cout << "  lower_tri_solve version" << std::endl;
          KokkosSparse::Impl::Experimental::lower_tri_solve(thandle, row_map, entries, values, rhs, lhs);
#endif
        }
        else {
          KokkosSparse::Impl::Experimental::upper_tri_solve(thandle, row_map, entries, values, rhs, lhs);
        }
      }
      else if ( thandle.get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1CHAIN ) {
        auto cutoff_threshold = thandle->get_chain_threshold();
        std::cout << "  lower_tri_solve cutoff: " << cutoff_threshold << std::endl;
        if (is_lower)
          KokkosSparse::Impl::Experimental::tri_solve_chain(thandle, row_map, entries, values, rhs, lhs, true);
        else
          KokkosSparse::Impl::Experimental::tri_solve_chain(thandle, row_map, entries, values, rhs, lhs, false);
      }
      else if( thandle->get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1 
               || thandle->get_algorithm() == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP2
             )
      {
        Kokkos::Timer timer;
//        KokkosSparse::Impl::Experimental::numeric_dense_partition_algm(thandle, row_map, entries, values); // vals not an argument, this needs its own file, or move to solve...
//        std::cout << "Numeric time: " << timer.seconds() << std::endl;
//        thandle->set_numeric_complete();
        timer.reset();
        if (is_lower)
          KokkosSparse::Impl::Experimental::tri_solve_partition_dense(thandle, row_map, entries, values, rhs, lhs, true);
        else
          KokkosSparse::Impl::Experimental::tri_solve_partition_dense(thandle, row_map, entries, values, rhs, lhs, false);

        std::cout << "Solve time: " << timer.seconds() << std::endl;
      }

}



} // namespace Experimental
} // namespace Impl
} // namespace KokkosSparse

#ifdef LVL_OUTPUT_INFO
#undef LVL_OUTPUT_INFO
#endif

#endif
