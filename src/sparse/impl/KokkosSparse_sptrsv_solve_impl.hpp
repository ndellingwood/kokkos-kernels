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

//#define LVL_OUTPUT_INFO
//#define CHAIN_DEBUG_OUTPUT
//#define TRISOLVE_TIMERS

#define KOKKOSPSTRSV_SOLVE_IMPL_PROFILE 1
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
#endif

struct UnsortedTag {};


// This functor unifies the lower and upper implementations, the hope is the "is_lower" check does not add noticable time on larger problems
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
  long node_groups;


  TriLvlSchedTP1SolverFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, const bool is_lowertri_, long node_count_, long node_groups_ = 0) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), is_lowertri(is_lowertri_), node_count(node_count_), node_groups(node_groups_) {}


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
  // team_size: each team can be assigned a row, if there are enough rows...


  LowerTriLvlSchedTP1SingleBlockFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, NGBLType &nodes_per_level_, long node_count_, long lvl_start_, long lvl_end_ ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), nodes_per_level(nodes_per_level_), node_count(node_count_), lvl_start(lvl_start_), lvl_end(lvl_end_) {}

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
  // team_size: each team can be assigned a row, if there are enough rows...


  UpperTriLvlSchedTP1SingleBlockFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, NGBLType &nodes_per_level_, long node_count_, long lvl_start_, long lvl_end_ ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), nodes_per_level(nodes_per_level_), node_count(node_count_), lvl_start(lvl_start_), lvl_end(lvl_end_) {}

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
};


struct LargerCutoffTag {};

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
  const bool is_lower;
  const int  cutoff;
  // team_size: each team can be assigned a row, if there are enough rows...


  TriLvlSchedTP1SingleBlockFunctor( const RowMapType &row_map_, const EntriesType &entries_, const ValuesType &values_, LHSType &lhs_, const RHSType &rhs_, const NGBLType &nodes_grouped_by_level_, NGBLType &nodes_per_level_, long node_count_, long lvl_start_, long lvl_end_, const bool is_lower_, const int cutoff_ = 0 ) :
    row_map(row_map_), entries(entries_), values(values_), lhs(lhs_), rhs(rhs_), nodes_grouped_by_level(nodes_grouped_by_level_), nodes_per_level(nodes_per_level_), node_count(node_count_), lvl_start(lvl_start_), lvl_end(lvl_end_), is_lower(is_lower_), cutoff(cutoff_) {}

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
          auto colid = entries(ptr);
          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
        if (is_lower)
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
      diff = scalar_t(0.0);
      // If cutoff > team_size, then a thread will be responsible for multiple rows - this may be a helpful scenario depending on occupancy etc.
      for (int my_rank = my_team_rank; my_rank < cutoff; my_rank+=team.team_size() ) {
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
          auto colid = entries(ptr);
          auto val   = values(ptr);
#ifdef CHAIN_DEBUG_OUTPUT
          printf("  ptr: %d  colid: %d  val: %lf  rank: %d\n", (int)ptr, (int)colid, val, team.team_rank());
#endif
          if ( colid != rowid ) {
            diff -= val*lhs(colid);
          }
        }
        // ASSUMPTION: sorted diagonal value located at eoffset - 1 for lower tri, soffset for upper tri
        if (is_lower)
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


// --------------------------------
// solver control routines
// --------------------------------

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

        //LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
        TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count);
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
          team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 128;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is responsible for processing team_size # nodes
        //       TeamThreadRange over number nodes of node_groups
        //       To avoid masking threads, 1 thread (team) per node in node_group (thread has full ownership of a node)
        //       ThreadVectorRange responsible for the actual solve computation
        const int node_groups = team_size;

        LowerTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count, node_groups);
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
  std::cout << "  (l)tri_solve_chain: setup = " << time_setup << std::endl;
  std::cout << "  (l)tri_solve_chain: total loop = " << time_outer << std::endl;
  std::cout << "  (l)tri_solve_chain: accum lvl solves = " << time_inner_total << std::endl;
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

//        UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
        TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, false, node_count);
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
          team_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 128;
        }
        int vector_size = thandle.get_team_size();
        if ( vector_size == -1 ) {
          vector_size = std::is_same< typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace >::value ? 1 : 4;
        }

        // This impl: "chunk" lvl_nodes into node_groups; a league_rank is responsible for processing that many nodes
        //       TeamThreadRange over number nodes of node_groups
        //       To avoid masking threads, 1 thread (team) per node in node_group (thread has full ownership of a node)
        //       ThreadVectorRange responsible for the actual solve computation
        const int node_groups = team_size;

        UpperTriLvlSchedTP2SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count, node_groups);
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
  std::cout << "  (u)tri_solve_chain: setup = " << time_setup << std::endl;
  std::cout << "  (u)tri_solve_chain: total loop = " << time_outer << std::endl;
  std::cout << "  (u)tri_solve_chain: accum lvl solves = " << time_inner_total << std::endl;
  std::cout << "     solve calls = " << tp1_ctr << std::endl;
  std::cout << "********************************" << std::endl; 
#endif

} // end upper_tri_solve


template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void tri_solve_chain(TriSolveHandle & thandle, const RowMapType row_map, const EntriesType entries, const ValuesType values, const RHSType & rhs, LHSType &lhs, const bool is_lower) {

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
  //auto hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
  //Kokkos::deep_copy(hnodes_per_level, nodes_per_level);

  auto nodes_grouped_by_level = thandle.get_nodes_grouped_by_level();

  size_type node_count = 0;
#ifdef TRISOLVE_TIMERS
  // prep time
  time_setup = timer_setup.seconds(); 
  timer_outer.reset();
#endif
  
  for ( size_type chainlink = 0; chainlink < num_chain_entries; ++chainlink ) {
    size_type schain = h_chain_ptr(chainlink);
    size_type echain = h_chain_ptr(chainlink+1);

  #ifdef TRISOLVE_TIMERS
     // fenced solve time
    timer_wrap_ifelse.reset();
  #endif
    if ( echain - schain == 1 ) {
      //std::cout << "Call regular single-link TP - chainlink: " << chainlink << std::endl;
      // run normal algm as this is a single level
      // schain should.... map to the level....
        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        int team_size = thandle.get_team_size();

        size_type lvl_nodes = hnodes_per_level(schain); //lvl == echain????
  #ifdef TRISOLVE_TIMERS
      // full-solve time
      tp1_ctr++;
      std::cout << "  *** Calling non-single-block solve *** " << std::endl;
      std::cout << "      team_size = " << team_size << std::endl;
      std::cout << "      lvl_nodes = " << lvl_nodes << std::endl;
      timer_full_solve.reset();
  #endif

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
        if (is_lower) {
          // TODO Time changes between merged functor and individuals
          //LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count);
          if ( team_size == -1 )
            Kokkos::parallel_for("parfor_l_team_chain1auto", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
          else
            Kokkos::parallel_for("parfor_l_team_chain1", policy_type( lvl_nodes , team_size ), tstf);
        }
        else {
          //UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, false, node_count);
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
        size_type lvl_nodes = 0;

        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        typedef Kokkos::TeamPolicy<LargerCutoffTag, execution_space> large_cutoff_policy_type;
        auto cutoff = thandle.get_chain_threshold();
        const int team_size = cutoff;
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
        if (is_lower) {
          if (cutoff <= team_size) {
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true);
//          LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
            Kokkos::parallel_for("parfor_l_team_chainmulti", policy_type( 1, team_size ), tstf);
          }
          else {
            // team_size < cutoff => kernel must allow for a block-stride internally
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true, cutoff);
//          LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
            Kokkos::parallel_for("parfor_l_team_chainmulti_cutoff", large_cutoff_policy_type( 1, team_size ), tstf);
          }
        }
        else {
          if (cutoff <= team_size) {
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, false);
//          UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
            Kokkos::parallel_for("parfor_u_team_chainmulti", policy_type( 1, team_size ), tstf);
          }
          else {
            // team_size < cutoff => kernel must allow for a block-stride internally
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, false, cutoff);
//          UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
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

} // end tri_solve_chain


#ifdef DENSEPARTITION
template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType, class RHSType, class LHSType >
void tri_solve_partition_dense(TriSolveHandle & thandle, const RowMapType frow_map, const EntriesType entries, const ValuesType values, const RHSType & frhs, LHSType & flhs, const bool is_lower) {

  typedef typename TriSolveHandle::execution_space execution_space;
  typedef typename TriSolveHandle::size_type size_type;
  typedef typename TriSolveHandle::nnz_lno_view_t NGBLType;

// Partitioned matrix lower tri solve
// 1. Solve the sparse portion as before - symbolic was already conducted using subview matrices for this
//    - Will require taking subviews of the input arrays
//    - fence()
// 2. gemv on dense block, using results from (1)
// 3. dense trisolve for remaining dense portion of x - use x as rhs and lhs in this step
//    - will require a cublas impl, or pre-computing L^-1 on host, copying, and applying as gemv


// Part 1. Sparse partition of the matrix, computation done as in other algorithms, just need to take subviews of the input view arrays
  auto dense_row_start = thandle.get_dense_partition_row_start();
  //auto dense_partition_nrows = thandle.get_dense_partition_nrows() ;

  //auto persist_sptrimtx_nrows = thandle.get_persist_sptrimtx_nrows();


  auto row_map = Kokkos::subview(frow_map, Kokkos::pair<size_type,size_type>(0, dense_row_start+1));
  // Need the offset into entries and vals; for sparse partition want the range [ 0, drow_map(dense_row_start) )
  // FIXME This shouldn't be needed
  //auto num_spentries = thandle.get_nnz_persist_spmtx();
  //auto entries = Kokkos::subview(fentries, Kokkos::pair<size_type,size_type>(0, num_spentries));
  // FIXME This shouldn't be needed
  //auto values = Kokkos::subview(fvalues, Kokkos::pair<size_type,size_type>(0, num_spentries));

  auto rhs = Kokkos::subview(frhs, Kokkos::pair<size_type,size_type>(0, dense_row_start));
  auto lhs = Kokkos::subview(flhs, Kokkos::pair<size_type,size_type>(0, dense_row_start));

  // upper versions:
  //auto row_map = Kokkos::subview(frow_map, Kokkos::pair<size_type,size_type>(dense_row_start,frow_map.extent(0)));
  //auto rhs = Kokkos::subview(frhs, Kokkos::pair<size_type,size_type>(dense_row_start, frhs.extent(0)));
  //auto lhs = Kokkos::subview(flhs, Kokkos::pair<size_type,size_type>(dense_row_start, flhs.extent(0)));

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStop();
#endif
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

  size_type node_count = 0;
#ifdef TRISOLVE_TIMERS
  // prep time
  time_setup = timer_setup.seconds(); 
  timer_outer.reset();
#endif
  
  for ( size_type chainlink = 0; chainlink < num_chain_entries; ++chainlink ) {
    size_type schain = h_chain_ptr(chainlink);
    size_type echain = h_chain_ptr(chainlink+1);

  #ifdef TRISOLVE_TIMERS
     // fenced solve time
    timer_wrap_ifelse.reset();
  #endif
    if ( echain - schain == 1 ) {
      //std::cout << "Call regular single-link TP - chainlink: " << chainlink << std::endl;
      // run normal algm as this is a single level
      // schain should.... map to the level....
      typedef Kokkos::TeamPolicy<execution_space> policy_type;
      int team_size = thandle.get_team_size();

      size_type lvl_nodes = hnodes_per_level(schain); //lvl == echain????
  #ifdef TRISOLVE_TIMERS
      // full-solve time
      tp1_ctr++;
      std::cout << "  *** Calling non-single-block solve *** " << std::endl;
      std::cout << "      team_size = " << team_size << std::endl;
      std::cout << "      lvl_nodes = " << lvl_nodes << std::endl;
      timer_full_solve.reset();
  #endif

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOSPSTRSV_SOLVE_IMPL_PROFILE)
cudaProfilerStart();
#endif
        if (is_lower) {
          // TODO Time changes between merged functor and individuals
          //LowerTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, true, node_count);
          if ( team_size == -1 )
            Kokkos::parallel_for("parfor_l_team_chain1auto", policy_type( lvl_nodes , Kokkos::AUTO ), tstf);
          else
            Kokkos::parallel_for("parfor_l_team_chain1", policy_type( lvl_nodes , team_size ), tstf);
        }
        else {
          //UpperTriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, node_count);
          TriLvlSchedTP1SolverFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, false, node_count);
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
        size_type lvl_nodes = 0;

        typedef Kokkos::TeamPolicy<execution_space> policy_type;
        typedef Kokkos::TeamPolicy<LargerCutoffTag, execution_space> large_cutoff_policy_type;
        auto cutoff = thandle.get_chain_threshold();
        const int team_size = cutoff;
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
        if (is_lower) {
          if (cutoff <= team_size) {
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true);
//          LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
            Kokkos::parallel_for("parfor_l_team_chainmulti", policy_type( 1, team_size ), tstf);
          }
          else {
            // team_size < cutoff => kernel must allow for a block-stride internally
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, true, cutoff);
//          LowerTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
            Kokkos::parallel_for("parfor_l_team_chainmulti_cutoff", large_cutoff_policy_type( 1, team_size ), tstf);
          }
        }
        else {
          if (cutoff <= team_size) {
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, false);
//          UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
            Kokkos::parallel_for("parfor_u_team_chainmulti", policy_type( 1, team_size ), tstf);
          }
          else {
            // team_size < cutoff => kernel must allow for a block-stride internally
            TriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain, false, cutoff);
//          UpperTriLvlSchedTP1SingleBlockFunctor<RowMapType, EntriesType, ValuesType, LHSType, RHSType, NGBLType> tstf(row_map, entries, values, lhs, rhs, nodes_grouped_by_level, nodes_per_level, node_count, schain, echain);
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


// Part 2. gemv, set xp <- bp - Mp*xknown
//                 lhsp <- rhsp - Mp*lhs  lhs the subview from part 1.
// Process:
//           1. lhsp = Kokkos::subview(flhs, pair(cutoff,nrows); rhsp = Kokkos::subview(frhs, pair(cutoff,nrows); deep_copy(lhsp, rhsp);
//           2. gemv("N", -1.0, dense_mtx, lhs, 1.0, lhsp); (where rhsp i.e. b was copied into lhsp, and lhs is the solution from part 1)
//           3. Kokkos::fence(); ?
  //auto dense_mtx = thandle.get_dense_mtx_partition(); // FIXME Need to remove and replace with subview components in sparse components
  auto dense_tri = thandle.get_dense_trimtx_partition();

  // lower tri
  auto lhsp = Kokkos::subview(flhs, Kokkos::pair<size_type, size_type>(dense_row_start, flhs.extent(0))); 
  auto rhsp = Kokkos::subview(frhs, Kokkos::pair<size_type, size_type>(dense_row_start, frhs.extent(0))); 
  // upper tri - FIXME This must happen at the start of the solve
  //auto lhsp = Kokkos::subview(flhs, Kokkos::pair<size_type, size_type>(0,dense_row_start)); 
  //auto rhsp = Kokkos::subview(frhs, Kokkos::pair<size_type, size_type>(0,dense_row_start)); 
  Kokkos::deep_copy(lhsp, rhsp);

  //auto row_map_rectspmtx = thandle.get_row_map_rectspmtx_partition();
  //Create KokkosSparse::CrsMatrix from modified row_map, entries, and values
  // This may require "shifting" the modified row_map, subview the entries and values and shift the entries array by subtracting out colid shift
  //KokkosSparse::spmv("N", -1.0, k_persist_rectspmtx, lhs, 1.0, lhsp); //(where rhsp i.e. b was copied into lhsp, and lhs is the solution from part 1)

// Part 3. dense trisolve for remaining dense portion of x - use x as rhs and lhs in this step
//           * Treat lhsp as partially updated rhsp, overwrite for final result

// cublass API to dense trsv
//cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
//                           cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)

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
  cublasFillMode_t uplo = is_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
  //cublasDiagType_t diag = CUBLAS_DIAG_UNIT;
  bool tri_is_lr = std::is_same<Kokkos::LayoutRight, typename TriSolveHandle::mtx_scalar_view_t::array_layout >::value;
  const int AST = tri_is_lr?dense_tri.stride(0):dense_tri.stride(1);
  LDA = AST == 0 ? 1 : AST;

  stat = cublasDtrsv(cublashandle, uplo, trans, diag, dense_tri.extent(0), dense_tri.data(), LDA, lhsp.data(), 1);
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
#endif


} // end tri_solve_partition_dense
#endif



} // namespace Experimental
} // namespace Impl
} // namespace KokkosSparse

#ifdef LVL_OUTPUT_INFO
#undef LVL_OUTPUT_INFO
#endif

#endif
