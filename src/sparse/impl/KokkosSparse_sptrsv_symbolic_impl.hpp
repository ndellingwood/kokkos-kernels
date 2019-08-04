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

//#define LVL_OUTPUT_INFO

// TODO Pass values array and store diagonal entries - should this always be done or optional?

namespace KokkosSparse {
namespace Impl {
namespace Experimental {

template < class TriSolveHandle, class NPLViewType >
void symbolic_chain_phase(TriSolveHandle &thandle, const NPLViewType &nodes_per_level) {

  typedef typename TriSolveHandle::size_type size_type;

  size_type level = thandle.get_num_levels();

  // Create the chain now
  // FIXME Implementations will need to be templated on exec space it seems...
 auto cutoff_threshold = thandle.get_chain_threshold();
 std::cout << "SYMB cutoff = " << cutoff_threshold << std::endl;
 thandle.print_algorithm();
 if ( thandle.algm_requires_symb_chain() ) {
  std::cout << "SYMB Call CHAIN version" << std::endl;
  auto h_chain_ptr = thandle.get_host_chain_ptr();
  h_chain_ptr(0) = 0;
  size_type chain_length = 0;
  size_type num_chain_entries = 0;
  int update_chain = 0;
  //const int cutoff = std::is_same<typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>::value ? 1 : 256; // TODO chain cutoff hard-coded to 256: make this a "threshold" parameter in the handle
  const int cutoff = cutoff_threshold;
  std::cout << "cutoff = " << cutoff << std::endl;
  for ( size_type i = 0; i < level; ++i ) {
    auto cnpl = nodes_per_level(i);
    std::cout << "incre chain_length  npl(" << i << ") = " << nodes_per_level(i) << std::endl;
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
      std::cout << "  nce = " << num_chain_entries << "  chain_length = " << chain_length << std::endl;
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
      std::cout << "  nce = " << num_chain_entries << "  chain_length = " << chain_length << std::endl;

      num_chain_entries += 1;
      h_chain_ptr(num_chain_entries) = h_chain_ptr(num_chain_entries-1) + 1;

      chain_length = 0; //reset
      update_chain = 0; //reset
    }
  }
  thandle.set_num_chain_entries(num_chain_entries);
  std::cout << "  num_chain_entries = " << thandle.get_num_chain_entries() << std::endl;
  for ( size_type i = 0; i < num_chain_entries+1; ++i )
  {
    std::cout << "chain_ptr(" << i << "): " << h_chain_ptr(i) << std::endl;
  }
 }
  // Usage:
  // for c in [0, num_chain_entries)
  //   s = h_chain_ptr(c); e = h_chain_ptr(c+1);
  //   num_levels_in_current_chain = e - s;
  //   if nlicc > 256
  //     call current_alg
  //   else
  //     call single_block(s,e)

} // end symbolic_chain_phase


template < class TriSolveHandle, class RowMapType, class EntriesType >
void lower_tri_symbolic ( TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries) {

 if ( thandle.algm_requires_symb_lvlsched() )
 {
  // Scheduling currently compute on host - need host copy of all views

  typedef typename TriSolveHandle::size_type size_type;

  typedef typename TriSolveHandle::nnz_lno_view_t  DeviceEntriesType;
  typedef typename TriSolveHandle::nnz_lno_view_t::HostMirror HostEntriesType;

  typedef typename TriSolveHandle::signed_nnz_lno_view_t DeviceSignedEntriesType;
  typedef typename TriSolveHandle::signed_nnz_lno_view_t::HostMirror HostSignedEntriesType;

  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

  size_type nrows = thandle.get_nrows();

  auto row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(row_map, drow_map);

  auto entries = Kokkos::create_mirror_view(dentries);
  Kokkos::deep_copy(entries, dentries);
  
  DeviceEntriesType dnodes_per_level = thandle.get_nodes_per_level();
  HostEntriesType nodes_per_level = Kokkos::create_mirror_view(dnodes_per_level);
  Kokkos::deep_copy(nodes_per_level, dnodes_per_level);

  DeviceEntriesType dnodes_grouped_by_level = thandle.get_nodes_grouped_by_level();
  HostEntriesType nodes_grouped_by_level = Kokkos::create_mirror_view(dnodes_grouped_by_level);
  Kokkos::deep_copy(nodes_grouped_by_level, dnodes_grouped_by_level);

  DeviceSignedEntriesType dlevel_list = thandle.get_level_list();
  HostSignedEntriesType level_list = Kokkos::create_mirror_view(dlevel_list);
  Kokkos::deep_copy(level_list, dlevel_list);

  HostSignedEntriesType previous_level_list( Kokkos::ViewAllocateWithoutInitializing("previous_level_list"), nrows );
  Kokkos::deep_copy( previous_level_list, signed_integral_t(-1) );

  auto diagonal_offsets = thandle.get_diagonal_offsets();
  // diagonal_offsets is uninitialized - deep_copy unnecessary at the beginning, only needed at the end
  auto hdiagonal_offsets = Kokkos::create_mirror_view(diagonal_offsets);

  // node 0 is trivially independent in lower tri solve, start with it in level 0
  size_type level = 0;
  auto starting_node = 0;
  level_list(starting_node) = 0;
  size_type node_count = 1; //lower tri: starting with node 0 already in level 0

  nodes_per_level(0) = 1;
  nodes_grouped_by_level(0) = starting_node;

  while (node_count < nrows) {

    for ( size_type row = 1; row < nrows; ++row ) { // row 0 already included
      if ( level_list(row) == -1 ) { // unmarked
        bool is_root = true;
        signed_integral_t ptrstart = row_map(row);
        signed_integral_t ptrend   = row_map(row+1);

        for (signed_integral_t offset = ptrstart; offset < ptrend; ++offset) {
          size_type col = entries(offset);
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

  std::cout << "  lower_tri_symbolic result: " << std::endl;
  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "node: " << i << "  level_list = " << level_list(i) << std::endl; }

  for ( size_type i = 0; i < level; ++i )
  { std::cout << "level: " << i << "  nodes_per_level = " << nodes_per_level(i) << std::endl; }

  for ( size_type i = 0; i < node_count; ++i )
  { std::cout << "i: " << i << "  nodes_grouped_by_level = " << nodes_grouped_by_level(i) << std::endl; }
#endif

  Kokkos::deep_copy(dnodes_grouped_by_level, nodes_grouped_by_level);
  Kokkos::deep_copy(dnodes_per_level, nodes_per_level);
  Kokkos::deep_copy(dlevel_list, level_list);
  Kokkos::deep_copy(diagonal_offsets, hdiagonal_offsets);
 }
} // end lower_tri_symbolic


template < class TriSolveHandle, class RowMapType, class EntriesType >
void upper_tri_symbolic ( TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries ) {

 if ( thandle.algm_requires_symb_lvlsched() )
 {
  // Scheduling currently compute on host - need host copy of all views

  typedef typename TriSolveHandle::size_type size_type;

  typedef typename TriSolveHandle::nnz_lno_view_t  DeviceEntriesType;
  typedef typename TriSolveHandle::nnz_lno_view_t::HostMirror HostEntriesType;

  typedef typename TriSolveHandle::signed_nnz_lno_view_t DeviceSignedEntriesType;
  typedef typename TriSolveHandle::signed_nnz_lno_view_t::HostMirror HostSignedEntriesType;

  typedef typename TriSolveHandle::signed_integral_t signed_integral_t;

  size_type nrows = thandle.get_nrows();

  auto row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(row_map, drow_map);

  auto entries = Kokkos::create_mirror_view(dentries);
  Kokkos::deep_copy(entries, dentries);
  
  DeviceEntriesType dnodes_per_level = thandle.get_nodes_per_level();
  HostEntriesType nodes_per_level = Kokkos::create_mirror_view(dnodes_per_level);
  Kokkos::deep_copy(nodes_per_level, dnodes_per_level);

  DeviceEntriesType dnodes_grouped_by_level = thandle.get_nodes_grouped_by_level();
  HostEntriesType nodes_grouped_by_level = Kokkos::create_mirror_view(dnodes_grouped_by_level);
  Kokkos::deep_copy(nodes_grouped_by_level, dnodes_grouped_by_level);

  DeviceSignedEntriesType dlevel_list = thandle.get_level_list();
  HostSignedEntriesType level_list = Kokkos::create_mirror_view(dlevel_list);
  Kokkos::deep_copy(level_list, dlevel_list);

  HostSignedEntriesType previous_level_list( Kokkos::ViewAllocateWithoutInitializing("previous_level_list"), nrows);
  Kokkos::deep_copy( previous_level_list, signed_integral_t(-1) );

  auto diagonal_offsets = thandle.get_diagonal_offsets();
  // diagonal_offsets is uninitialized - deep_copy unnecessary at the beginning, only needed at the end
  auto hdiagonal_offsets = Kokkos::create_mirror_view(diagonal_offsets);

  // final row is trivially independent in upper tri solve, start with it in level 0
  size_type level = 0;
  auto starting_node = nrows - 1;
  level_list(starting_node) = 0;
  size_type node_count = 1; //upper tri: starting with node n already in level 0

  nodes_per_level(0) = 1;
  nodes_grouped_by_level(0) = starting_node;

  while (node_count < nrows) {

    for ( signed_integral_t row = nrows-2; row >= 0; --row ) { // row 0 already included
      if ( level_list(row) == -1 ) { // unmarked
        bool is_root = true;
        signed_integral_t ptrstart = row_map(row);
        signed_integral_t ptrend   = row_map(row+1);

        for (signed_integral_t offset = ptrend-1; offset >= ptrstart; --offset) {
          signed_integral_t col = entries(offset);
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

  Kokkos::deep_copy(dnodes_grouped_by_level, nodes_grouped_by_level);
  Kokkos::deep_copy(dnodes_per_level, nodes_per_level);
  Kokkos::deep_copy(dlevel_list, level_list);
  Kokkos::deep_copy(diagonal_offsets, hdiagonal_offsets);
 }
} // end upper_tri_symbolic



} // namespace Experimental
} // namespace Impl
} // namespace KokkosSparse

#endif
