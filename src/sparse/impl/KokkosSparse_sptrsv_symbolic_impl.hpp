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

#define LVL_OUTPUT_INFO
#define CHAIN_LVL_OUTPUT_INFO

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

} // end symbolic_chain_phase





template < class TriSolveHandle, class RowMapType, class EntriesType >
void lower_tri_symbolic (TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries) {

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

  // node 0 is trivially independent in lower tri solve, start with it in level 0
  size_type level = 0;
  // FIXME Change this to allow for partitioned sparse mtx
#ifdef DENSEPARTITION
  auto starting_node = thandle.get_lvlsched_node_start();
  auto ending_node = thandle.get_lvlsched_node_end();
#else
  auto starting_node = 0;
  auto ending_node = nrows
#endif

  level_list(starting_node) = level;
  size_type node_count = 1; //lower tri: starting with node 0 already in level 0

  nodes_per_level(0) = 1;
  nodes_grouped_by_level(0) = starting_node;

  while (node_count < nrows) {

    for ( size_type row = starting_node+1; row < ending_node; ++row ) { // row 0 already included
      if ( level_list(row) == -1 ) { // unmarked
        bool is_root = true;
        signed_integral_t ptrstart = row_map(row);
        signed_integral_t ptrend   = row_map(row+1);

        for (signed_integral_t offset = ptrstart; offset < ptrend; ++offset) {
          size_type col = entries(offset);
          // FIXME: For lower_tri, colid is unchanged; shifted for upper_tri...
          // FIXME: This check for col != row vs col < row will be broken with partial persist_sptrimtx, since using a row_map with rowid shifted to 0 but not colid
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

  // Deep copy to device views
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

  // final row is trivially independent in upper tri solve, start with it in level 0
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

  level_list(starting_node) = level;
  size_type node_count = 1; //upper tri: starting with node n already in level 0

  nodes_per_level(0) = 1;
  nodes_grouped_by_level(0) = starting_node;

  while (node_count < nrows) {

    for ( signed_integral_t row = starting_node-1; row >= ending_node; --row ) { // row 0 already included
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
} // end upper_tri_symbolic




#ifdef DENSEPARTITION
template < class TriSolveHandle, class RowMapType, class EntriesType >
void symbolic_dense_partition_algm( TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries) {

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

  // TODO Will the reference count help this allocation to persist beyond this function call?
  thandle.alloc_row_map_rectspmtx(dense_partition_nrows+1);
  auto row_map_rectspmtx = thandle.get_row_map_rectspmtx();
  //row_map_rectspmtx = typename TriSolveHandle::nnz_row_view_t("row_map_rectspmtx", dense_partition_nrows+1);
  std::cout << "  row_map_rectspmtx allocated? extent(0) = " << row_map_rectspmtx.extent(0) << std::endl;
  std::cout << "  row_map_rectspmtx.data() = " << row_map_rectspmtx.data() << std::endl;


  typedef Kokkos::View<size_type, typename RowMapType::memory_space> ShiftedEntriesStart;

  auto dense_row_start = thandle.get_dense_partition_row_start();

  std::cout << "\n\n  Begin symbolic dense partition" << std::endl;
  std::cout << "  dense_row_start = " << dense_row_start << std::endl;
  std::cout << "  dense_nrows = " << dense_partition_nrows << std::endl;

  ShiftedEntriesStart shifted_entries_start(Kokkos::ViewAllocateWithoutInitializing("ses"));

  auto dprow_map = Kokkos::subview( drow_map, Kokkos::pair<size_type,size_type>(dense_row_start, dense_row_start+dense_partition_nrows+1) );
  std::cout << "  dprow_map allocated? extent(0) = " << dprow_map.extent(0) << std::endl;
  std::cout << "  dprow_map.data() = " << dprow_map.data() << std::endl;
/*
  Kokkos::parallel_for("shifted_dense_row_map_init", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, row_map_rectspmtx.extent(0)),
    KOKKOS_LAMBDA (const size_type i) {
      if ( i == 0 )
        shifted_entries_start() = dprow_map(0);

      row_map_rectspmtx(i) = dprow_map(i) - dprow_map(0);
    });
*/
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

  std::cout << "  is_lower_tri = " << is_lower_tri << std::endl;
  std::cout << "  rectspmtx_col_start = " << rectspmtx_col_start << std::endl;
  std::cout << "  trimtx_col_start = " << trimtx_col_start << std::endl;

  // reshape CRS
  // dprow_map has proper offsets into dentries and dvalues
/*
  Kokkos::parallel_for("shifted_dense_row_map_init", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dprow_map.extent(0)),
    KOKKOS_LAMBDA (const size_type i) {
      size_type offset_start = dprow_map(i);
      size_type offset_end   = dprow_map(i+1);
      for (size_type offset = offset_start; offset < offset_end; ++offset) {
        size_type colid = dentries(offset);
        // Remove out-of-sparse-rect indices
        if ( (is_lower_tri && colid >= trimtx_col_start) || (!is_lower_tri && colid < rectspmtx_col_start) ) {
          // Decrement row_map_rectspmtx(i+1), but need to also decrement for every following index as well...
        }
      }
    });
*/
  // Frequency count of indices in rectspmtx of partition
  //Kokkos::parallel_for("row_map_rectspmtx freq count", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dprow_map.extent(0)),
  Kokkos::parallel_for("row_map_rectspmtx freq count", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, dense_partition_nrows),
    KOKKOS_LAMBDA (const size_type i) {
      size_type offset_start = dprow_map(i);
      size_type offset_end   = dprow_map(i+1);
      printf("offset_start = %d  offset_end = %d\n", offset_start, offset_end);
      for (size_type offset = offset_start; offset < offset_end; ++offset) {
        size_type colid = dentries(offset);
        printf("i = %d  colid = %d  is_lower_tri = %d\n",i, colid, (int)is_lower_tri);
        // Count in-sparse-rect entries per row, store at row_map(rowid+1) in anticipation of followup scan
        if ( (is_lower_tri && colid < trimtx_col_start) || (!is_lower_tri && colid >= rectspmtx_col_start) ) {
          // increment row_map_rectspmtx(i+1)
          ++row_map_rectspmtx(i+1);
        }
      }
    });
  Kokkos::fence();
  auto hrmrect = Kokkos::create_mirror_view(row_map_rectspmtx);
  Kokkos::deep_copy(hrmrect, row_map_rectspmtx);
  for (size_t i = 0; i < hrmrect.extent(0); ++i) {
    std::cout << "  freq hrmrect(" << i << ") = " << hrmrect(i) << std::endl;
  }

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
  Kokkos::deep_copy(hrmrect, row_map_rectspmtx);
  for (size_t i = 0; i < hrmrect.extent(0); ++i) {
    std::cout << "  hrmrect(" << i << ") = " << hrmrect(i) << std::endl;
  }

  thandle.set_nnz_rectspmtx(rectspmtx_nnz);

  std::cout << "  rectspmtx_nnz = " << rectspmtx_nnz << std::endl;

  thandle.alloc_entries_rectspmtx(rectspmtx_nnz); 
  auto entries_rectspmtx= thandle.get_entries_rectspmtx(); 
  //entries_rectspmtx= typename TriSolveHandle::nnz_lno_view_t("entries_rectspmtx", rectspmtx_nnz);
  std::cout << "  entries_rectspmtx allocated? extent(0) = " << entries_rectspmtx.extent(0) << std::endl;
  std::cout << "  entries_rectspmtx.data() = " << entries_rectspmtx.data() << std::endl;

  thandle.alloc_values_rectspmtx(rectspmtx_nnz);
  auto values_rectspmtx= thandle.get_values_rectspmtx();
  //values_rectspmtx= typename TriSolveHandle::nnz_scalar_view_t("values_rectspmtx", rectspmtx_nnz);
  std::cout << "  values_rectspmtx allocated? extent(0) = " << values_rectspmtx.extent(0) << std::endl;
  std::cout << "  values_rectspmtx.data() = " << values_rectspmtx.data() << std::endl;

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
          // increment row_map_rectspmtx(i+1)
          //++row_map_rectspmtx(i+1);
          entries_rectspmtx(new_idx_offset) = is_lower_tri ? colid : colid - dense_partition_nrows;
          ++new_idx_offset;
        }
      }

    });

  Kokkos::fence();
  auto hentrect = Kokkos::create_mirror_view(entries_rectspmtx);
  Kokkos::deep_copy(hentrect, entries_rectspmtx);
  for (size_t i = 0; i < hentrect.extent(0); ++i) {
    std::cout << "  freq hentrect(" << i << ") = " << hentrect(i) << std::endl;
  }
  
  // alloc dense tri mtx
  thandle.alloc_dense_trimtx(dense_partition_nrows, dense_partition_nrows);
  auto dense_trimtx= thandle.get_dense_trimtx();
  //dense_trimtx= typename TriSolveHandle::mtx_scalar_view_t("dense_tri_mtx", dense_partition_nrows, dense_partition_nrows);
  std::cout << "  dense_trimtx allocated? extent(0) = " << dense_trimtx.extent(0) << "  extent(1) = " << dense_trimtx.extent(1) << std::endl;
  std::cout << "  dense_trimtx.data() = " << dense_trimtx.data() << std::endl;

#if 0
  // FIXME BELOW IS OLD STUFF, may have useful indexing to not have to rethink but will delete soon...
  //
  // TODO Should I have the outer/public symbolic phase call this routine for the partition-type algorithms, create subviews and then call
  // the level scheduling based routines?

  // TODO Where should this partition_phase be called? I will reuse the existing symbolic routines on the sparse partition of the matrix,
  // but I intend to use subviews for that
  // TODO Allocate dense mtx and tri here, or in the handle initialization? Will need to reset if symbolic is called again with different params, but the handle should be reset in that case anyway...

  // TODO Symbolic should be called once - determines level_list, nodes_grouped_by_level, nodes_per_level for a fixed structure; allocates the views for dense matrix copies
  //          subviews for the sparse partitions will need to be created here to determine data mentioned above
  //          symbolic does not use the CRS vals array - so a change of the values but not structure does not require recalling symbolic
  //
  //      Current sptrsv requires:
  //        create handle
  //          stores lower vs upper, algm choice, nrows, algm-specific views (level_list, host_chain_ptr, etc)
  //
  //        sptrsv_symbolic(handle, rowmap, entries);
  //          analysis stores data for level schedule, chain_ptr, etc
  //          dense partition: allocate
  //
  //        sptrsv_numeric(handle, rowmap, entries, vals);
  //          populate the dense partition views (on device) stored in handle, 
  //
  //        sptrsv_solve(handle, rowmap, entries, vals, b, x)
  //        TODO Add interface taking CrsMatrix input?
  //          Reuse: simply pass different vals View
  //          dense partition: Call sptrsv_solve on sparse partition subviews; fence; gemv on dense mtx, overwrite x; dense trsv on dense tri, overwritten x subview as rhs b, re-overwrite subview of x for final result
  //
  //        destroy handle
  //
  //

  auto dense_start_row = thandle.get_dense_partition_row_start();

  // FIXME Inefficient way to get this value, do this in the handle? Otherwise, store it in the handle?
  auto h_row_map = Kokkos::create_mirror_view(drow_map);
  Kokkos::deep_copy(h_row_map, drow_map);
  auto num_spentries = h_row_map(dense_start_row); // number of items (nnz) from the entries array in the sparse partition

  thandle.set_nnz_persist_sptrimtx(num_spentries);

  // Create persisting sparse partition of the matrix - subview of row_map with offsets into original entries and values should suffice
  // Still starting at row 0, but ending at rowid < nrows - 1
  // TODO Check if above statment holds - are assumptions about starting nodes during level scheduling broken???
  auto sprow_map = Kokkos::subview(drow_map, Kokkos::pair<size_type,size_type>(0, dense_start_row+1));
  // Need the offset into entries and vals; for sparse partition want the range [ 0, drow_map(dense_start_row) )
  auto spentries = Kokkos::subview(dentries, Kokkos::pair<size_type,size_type>(0, num_spentries));
  if (thandle.is_lower_tri()) {
    lower_tri_symbolic(thandle, sprow_map, spentries);
  }
  else {
    upper_tri_symbolic(thandle, sprow_map, spentries);
  }
#else
  auto sptrimtx_row_start = thandle.get_persist_sptrimtx_row_start();
  auto sptrimtx_nrows = thandle.get_persist_sptrimtx_nrows();
  auto sptrimtx_row_map = Kokkos::subview( drow_map, Kokkos::pair<size_type,size_type>(sptrimtx_row_start, sptrimtx_row_start+sptrimtx_nrows+1) );

  if (thandle.is_lower_tri()) {
    lower_tri_symbolic(thandle, sptrimtx_row_map, dentries);
  }
  else {
    upper_tri_symbolic(thandle, sptrimtx_row_map, dentries);
  }
#endif
}


template < class TriSolveHandle, class RowMapType, class EntriesType, class ValuesType >
void numeric_dense_partition_algm(TriSolveHandle &thandle, const RowMapType drow_map, const EntriesType dentries, const ValuesType dvalues) {

  typedef typename TriSolveHandle::size_type size_type;

  std::cout << "\n\n  Begin numeric" << std::endl;
  auto dense_partition_nrows = thandle.get_dense_partition_nrows() ;
  auto dense_row_start = thandle.get_dense_partition_row_start();

  auto dprow_map = Kokkos::subview( drow_map, Kokkos::pair<size_type,size_type>(dense_row_start, dense_row_start+dense_partition_nrows+1) );

  //size_type rectsptmx_nnz = 0;
  size_type rectspmtx_col_start;
  size_type trimtx_col_start;

  auto row_map_rectspmtx = thandle.get_row_map_rectspmtx();
  // Broken here - row_map_rectspmtx in symbolic did not persist...
  std::cout << "  row_map_rectspmtx allocated? extent(0) = " << row_map_rectspmtx.extent(0) << std::endl;
  std::cout << "  row_map_rectspmtx.data() = " << row_map_rectspmtx.data() << std::endl;

  bool is_lower_tri = thandle.is_lower_tri();

  if (is_lower_tri) {
    rectspmtx_col_start = 0; // ends at trimtx_col_start
    trimtx_col_start = dense_row_start; // ends at nrows
  }
  else {
    rectspmtx_col_start = dense_partition_nrows; // ends at nrows
    trimtx_col_start = 0; // ends at rectspmtx_col_start
  }

  std::cout << "  is_lower_tri = " << is_lower_tri << std::endl;
  std::cout << "  rectspmtx_col_start = " << rectspmtx_col_start << std::endl;
  std::cout << "  trimtx_col_start = " << trimtx_col_start << std::endl;


  auto dense_trimtx= thandle.get_dense_trimtx();

  auto values_rectspmtx= thandle.get_values_rectspmtx();
  std::cout << "  values_rectspmtx allocated? extent(0) = " << values_rectspmtx.extent(0) << std::endl;
  std::cout << "  values_rectspmtx.data() = " << values_rectspmtx.data() << std::endl;

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
  auto hvrs = Kokkos::create_mirror_view(values_rectspmtx);
  Kokkos::deep_copy(hvrs, values_rectspmtx);
  for (size_t i = 0; i < hvrs.extent(0); ++i) {
    std::cout << "  hvrs(" << i << ") = " << hvrs(i) << std::endl;
  }

  Kokkos::fence();
  auto hdense_tri = Kokkos::create_mirror_view(dense_trimtx);
  Kokkos::deep_copy(hdense_tri, dense_trimtx);
  for (size_t i = 0; i < hdense_tri.extent(0); ++i) {
    for (size_t j = 0; j < hdense_tri.extent(1); ++j) {
      std::cout << "  hdense_tri(" << i << "," << j << ") = " << hdense_tri(i,j) << std::endl;
    }
  }


  std::cout << "  numeric complete" << std::endl;
  

// OLD STUFF

#if 0

  // Current:
  // handle: allocates views for the "dense" partitions
  // symb creates subviews of "sparse" partition then runs lvl scheduling using them
  // num  fills the "dense" partitions; requires retrieving from the row_map the shifted start into the entries array 
  //
  // Update:
  // handle: Add "subviews" for the sparse partitions, sparse-dense partitions, dense tri matrix, store start_offset into entries
  // symb:   allocate dense tri subview, sparse partition subviews
  // num:    subviews for dense partition spmv matrix
  // solve:  retrieve matrices above, replace gemv with spmv


  // Move this stuff to symbolic
  typedef Kokkos::View<size_type, typename RowMapType::memory_space> ShiftedEntriesStart;
  typedef Kokkos::View<typename RowMapType::non_const_value_type, typename RowMapType::array_layout, typename RowMapType::memory_space> MngRowMapType;

  auto dense_start_row = thandle.get_dense_partition_row_start();

  std::cout << "  Begin numeric" << std::endl;
  std::cout << "  dense_start_row = " << dense_start_row << std::endl;

  ShiftedEntriesStart shifted_entries_start(Kokkos::ViewAllocateWithoutInitializing("ses"));

  auto dprow_map = Kokkos::subview(drow_map, Kokkos::pair<size_type,size_type>(dense_start_row, drow_map.extent(0)));
  //auto shifted_dense_row_map = Kokkos::create_mirror(dprow_map);
  MngRowMapType shifted_dense_row_map("shifted_dense_row_map", drow_map.extent(0));

  Kokkos::parallel_for("shifted_dense_row_map_set", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, shifted_dense_row_map.extent(0)),
    KOKKOS_LAMBDA (const size_type i) {
      if ( i == 0 )
        shifted_entries_start() = dprow_map(0);

      shifted_dense_row_map(i) = dprow_map(i) - dprow_map(0);
    });

// [ x 0 0 0 0
//   0 x x 0 0
//   0 0 x x 0
//   0 0 0 x x
//   0 0 0 0 x ]

// Rows are shifted - "cut off" the upper sparse partition
// Colids stay the same, but need to offset into idx properly
//
// Example
// nrows = 5
// ptr: 0 1 3 5 7 8
// idx: 0 | 1 2 | 2 3 | 3 4 | 4
// nnz_per_row: 1 2 2 2 1
// shift: start at row 3
// partial_ptr (via subview): 5 7 8
// offset idx ignore entries up to ptr(shift) == ptr(3) == 5 == partial_ptr(0), the new start; end == idx.extent(0) == 8
// offset_idx: x | x x | x x | 3 4 | 4
// shifted_ptr sub 5: 0 2 3
// shifted_idx (via subview):  3 4 | 4


  // Very inefficient, but need the offset?
  //auto h_row_map = Kokkos::create_mirror_view(drow_map);
  //Kokkos::deep_copy(h_row_map, drow_map);

  auto h_shifted_entries_start_view = Kokkos::create_mirror_view(shifted_entries_start);
  Kokkos::deep_copy(h_shifted_entries_start_view, shifted_entries_start);
  //RowMapType::value_type h_shifted_entries_start;
  //Kokkos::deep_copy(h_shifted_entries_start, shifted_entries_start);

  typename RowMapType::value_type h_shifted_entries_end = drow_map.extent(0);

  typename TriSolveHandle::nnz_lno_unmanaged_view_t shifted_dense_entries = Kokkos::subview(dentries, Kokkos::pair<size_type,size_type>(h_shifted_entries_start_view(), h_shifted_entries_end)); // this is for "dense" portion

  //auto shifted_dense_entries = Kokkos::subview(dentries, Kokkos::pair<size_type,size_type>(h_shifted_entries_start_view(), h_shifted_entries_end)); // this is for "dense" portion

  typename TriSolveHandle::nnz_scalar_unmanaged_view_t shifted_dense_values  = Kokkos::subview(dvalues, Kokkos::pair<size_type,size_type>(h_shifted_entries_start_view(), h_shifted_entries_end)); // this is for "dense" portion
  //auto shifted_dense_values  = Kokkos::subview(dvalues, Kokkos::pair<size_type,size_type>(h_shifted_entries_start_view(), h_shifted_entries_end)); // this is for "dense" portion

  std::cout << "  numeric subview types" << std::endl;
  std::cout << "    dentries type: " << typeid(dentries).name() << std::endl;
  std::cout << "    shifted_dense_entries type: " << typeid(shifted_dense_entries).name() << std::endl;
  std::cout << "    dvalues type: " << typeid(dvalues).name() << std::endl;
  std::cout << "    shifted_dense_values type: " << typeid(shifted_dense_values).name() << std::endl;

  // FIXME TODO Use the shifted sparse matrix components just created above for the spmv rather than the dense gemv in the solve

  // OLD
  //auto cp_entries_start = h_row_map(dense_start_row+1);
  //auto cp_entries_end = h_row_map(drow_map.extent(0));

  // Need the offset into entries and vals; for dense partition want the range [ drow_map(dense_start_row+1), end )
  //auto cpentries = Kokkos::subview(dentries, Kokkos::pair<size_type,size_type>(cp_entries_start, cp_entries_end)); // this is for dense portion
  //auto cpvals = Kokkos::subview(dvalues, Kokkos::pair<size_type,size_type>(cp_entries_start, cp_entries_end)); // this is for dense portion

  // Fill the dense_mtx
  auto dense_mtx = thandle.get_dense_mtx_partition();
  auto dense_tri = thandle.get_dense_trimtx();

  // TODO If resetting the matrix values, but structure the same, need to re-zero the dense matrices
  // TODO Zero the matrices here, and alloc without initializing in handle (or symbolic)?
  Kokkos::deep_copy(dense_mtx, 0);
  Kokkos::deep_copy(dense_tri, 0);

  auto dense_mtx_rows = dense_mtx.extent(0);
  auto dense_mtx_cols = dense_mtx.extent(1);

//  auto num_tri_rows = dense_tri.extent(0);
//  auto num_tri_cols = dense_tri.extent(1);


  // Simple range_policy to parallelize over rows; can try out team_policy, assigning team to row and coordinating threads over the writes since matrix is dense, no conflicts
  // that may expose more parallelism
  Kokkos::parallel_for("numeric fill dense matrices", Kokkos::RangePolicy<typename TriSolveHandle::execution_space>(0, dense_mtx_rows), KOKKOS_LAMBDA(const size_type rowid) {
    // Assume entries and vals are ordered...
    auto offset_begin = dprow_map(rowid); // this is using the shifted rowid in the subview of row_map; the offset returned is that for the ORIGINAL entries and vals
    auto offset_end   = dprow_map(rowid+1);
    for (size_type offset = offset_begin; offset < offset_end; ++offset) {
      size_t colid = dentries(offset); // original global colid - use this for dense_mtx, but map it to start at 0 for the dense_tri matrix
      if (colid < dense_mtx_cols) {
        dense_mtx(rowid,colid) = dvalues(offset);
      }
      else {
        dense_tri(rowid, colid-dense_start_row) = dvalues(offset);
      }
    }
  });

#endif

  thandle.set_numeric_complete();
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
