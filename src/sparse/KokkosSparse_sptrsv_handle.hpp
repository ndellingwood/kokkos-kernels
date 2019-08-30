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

#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

#ifndef KOKKOSSPARSE_SPTRSVHANDLE_HPP
#define KOKKOSSPARSE_SPTRSVHANDLE_HPP

#define DENSEPARTITION

namespace KokkosSparse {
namespace Experimental {

// TODO TP2 algorithm had issues with some offset-ordinal combo to be addressed when compiled in Trilinos...
enum class SPTRSVAlgorithm { SEQLVLSCHD_RP, SEQLVLSCHD_TP1, SEQLVLSCHED_TP2, SEQLVLSCHD_TP1CHAIN , SEQLVLSCHD_DENSEP_TP1 };

template <class size_type_, class lno_t_, class scalar_t_,
          class ExecutionSpace,
          class TemporaryMemorySpace,
          class PersistentMemorySpace>
class SPTRSVHandle {
public:

  typedef ExecutionSpace HandleExecSpace;
  typedef TemporaryMemorySpace HandleTempMemorySpace;
  typedef PersistentMemorySpace HandlePersistentMemorySpace;

  typedef ExecutionSpace execution_space;
  typedef HandlePersistentMemorySpace memory_space;


  typedef typename std::remove_const<size_type_>::type  size_type;
  typedef const size_type const_size_type;

  typedef typename std::remove_const<lno_t_>::type  nnz_lno_t;
  typedef const nnz_lno_t const_nnz_lno_t;

  typedef typename std::remove_const<scalar_t_>::type  scalar_t;
  typedef const scalar_t const_nnz_scalar_t;


  // row_map type (managed memory)
  typedef typename Kokkos::View<size_type *, HandleTempMemorySpace> nnz_row_view_temp_t;
  typedef typename Kokkos::View<size_type *, HandlePersistentMemorySpace> nnz_row_view_t;
  typedef typename nnz_row_view_t::HostMirror host_nnz_row_view_t;
 // typedef typename row_lno_persistent_work_view_t::HostMirror row_lno_persistent_work_host_view_t; //Host view type
  typedef typename Kokkos::View<const size_type *, HandlePersistentMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess>> nnz_row_unmanaged_view_t; // for rank1 subviews

  // values type (managed memory)
  typedef typename Kokkos::View<scalar_t *, HandleTempMemorySpace> nnz_scalar_view_temp_t;
  typedef typename Kokkos::View<scalar_t *, HandlePersistentMemorySpace> nnz_scalar_view_t;
  typedef typename nnz_scalar_view_t::HostMirror host_nnz_scalar_view_t;
  typedef typename Kokkos::View<const scalar_t *, HandlePersistentMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess>> nnz_scalar_unmanaged_view_t; // for rank1 subviews

  // entries type (managed memory)
  typedef typename Kokkos::View<nnz_lno_t *, HandleTempMemorySpace> nnz_lno_view_temp_t;
  typedef typename Kokkos::View<nnz_lno_t *, HandlePersistentMemorySpace> nnz_lno_view_t;
  typedef typename nnz_lno_view_t::HostMirror host_nnz_lno_view_t;
  typedef typename Kokkos::View<const nnz_lno_t *, HandlePersistentMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess>> nnz_lno_unmanaged_view_t; // for rank1 subviews
 // typedef typename nnz_lno_persistent_work_view_t::HostMirror nnz_lno_persistent_work_host_view_t; //Host view type


  typedef typename std::make_signed<typename nnz_row_view_t::non_const_value_type>::type signed_integral_t;
  typedef Kokkos::View< signed_integral_t*, typename nnz_row_view_t::array_layout, typename nnz_row_view_t::device_type, typename nnz_row_view_t::memory_traits > signed_nnz_lno_view_t;
  typedef typename signed_nnz_lno_view_t::HostMirror host_signed_nnz_lno_view_t;

  typedef typename Kokkos::View<scalar_t **, HandlePersistentMemorySpace> mtx_scalar_view_t;

private:

  size_type nrows;

  bool lower_tri;

  SPTRSVAlgorithm algm;

  // Symbolic: Level scheduling data
  signed_nnz_lno_view_t level_list;
  nnz_lno_view_t nodes_per_level;
  host_nnz_lno_view_t hnodes_per_level; // NEW
  nnz_lno_view_t nodes_grouped_by_level;
  host_nnz_lno_view_t hnodes_grouped_by_level; // NEW
  size_type nlevel;

  int team_size;
  int vector_size;

  // TODO Store diagonal offsets
  nnz_lno_view_t diagonal_offsets;

  // Symbolic: Single-block chain data
  host_signed_nnz_lno_view_t h_chain_ptr;
  size_type num_chain_entries;
  signed_integral_t chain_threshold;



  bool symbolic_complete;
  bool require_symbolic_lvlsched_phase;
  bool require_symbolic_chain_phase;
  // TODO May be helpful to track completion of phases the full symbolic does not need to be repeated for chain_threshold changes, for example
//  bool symbolic_lvlsched_phase_complete;
//  bool symbolic_chain_phase_complete;

  void set_if_algm_require_symb_lvlsched () {
    if (algm == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP
        || algm == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1
        || algm == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHED_TP2
        || algm == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1CHAIN
#ifdef DENSEPARTITION
        || algm == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1
#endif
       )
    {
      require_symbolic_lvlsched_phase = true;
    }
    else {
      require_symbolic_lvlsched_phase = false;
    }
  }

  void set_if_algm_require_symb_chain () {
    if (algm == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1CHAIN
#ifdef DENSEPARTITION
        || algm == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1
#endif
       )
    {
      require_symbolic_chain_phase = true;
    }
    else {
      require_symbolic_chain_phase = false;
    }
  }


  // Symbolic and Numeric: Dense-block data structures
#ifdef DENSEPARTITION
  // TENTATIVE NAMING CONVENTIONS:
  //   persist_sptrimtx - the sparse tri matrix component remaining after partitioning
  //   rectspmtx - the sparse tri matrix component remaining after partitioning
  //   dense_partition* - refers to the partition with the sparse rect mtx and the dense triangular matrix after partitioning
  //
  // ASSUMPTIONS:
  //   lower_tri - the region [0, cutoff) is the sparse region, [cutoff, nrows) the "dense" partition
  //   upper_tri - the region [cutoff, nrows) is the sparse region, [0, cutoff) the "dense" partition
  //
  // KokkosBlas::gemv and cublas:trsv expect rank2 views/data for the matrix
  // Starting row for the dense partition, which will have a sparse rectangular matrix and dense triangular matrix

  // These two variables, with nrows, also determine that row_start and rows for the persist_sptrimtx (assumption differs for upper vs lower tri)
  size_type dense_partition_row_start;
  // This is set in the handle
  size_type dense_partition_nrows;

  bool auto_set_dense_partition_percent;
  float dense_partition_row_percent;

  // nnz for persisting sparse matrix after partition, equivalent to row_map(dense_partition_row_start); also provides offset into start of entries for the "dense" partition
  size_type nnz_persist_sptrimtx;

  size_type nnz_rectspmtx;


  mtx_scalar_view_t dense_trimtx; // 2D rectangular storage is inefficient for this matrix, but fits cuBLAS pattern...

  // needed for the rectspmtx partition to work properly with spmv routines
  nnz_row_view_t row_map_rectspmtx;

  nnz_lno_view_t entries_rectspmtx;

  nnz_scalar_view_t values_rectspmtx;

  //mtx_scalar_view_t dense_trimtx_inv_partition; // 2D storage is inefficient for this matrix, but fits cuBLAS pattern...

  //nnz_row_view_t    sparse_rowmap_partition; // Make this a subview of the original view - TODO Unneccessary to add here, just create subview before fcn call?
  //nnz_lno_view_t    sparse_entries_partition; // Make this a subview of the original view - TODO Unneccessary to add here, just create subview before fcn call?
  //nnz_scalar_view_t sparse_vals_partition; // Make this a subview of the original view - TODO Unneccessary to add here, just create subview before fcn call?
  //nnz_scalar_view_t sparse_bvector_partition; // Make this a subview of the original view - TODO Unneccessary to add here, just create subview before fcn call?

  bool require_symbolic_numeric_dense_phase;

  bool numeric_complete;

  void set_if_algm_require_dense_partition() {
    if (algm == KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1) {
      require_symbolic_numeric_dense_phase = true;
    }
    else {
      require_symbolic_numeric_dense_phase = false;
    }
  }

#endif


public:

  SPTRSVHandle(SPTRSVAlgorithm choice, const size_type nrows_, bool lower_tri_, bool symbolic_complete_ = false) :
    nrows(nrows_),
    lower_tri(lower_tri_),
    algm(choice),
    level_list(),
    nodes_per_level(),
    hnodes_per_level(),
    nodes_grouped_by_level(),
    hnodes_grouped_by_level(),
    nlevel(0),
    team_size(-1),
    vector_size(-1),
    diagonal_offsets(),
    h_chain_ptr(),
    num_chain_entries(0),
    chain_threshold(-1),
    symbolic_complete(symbolic_complete_),
    require_symbolic_lvlsched_phase(false),
    require_symbolic_chain_phase(false)
#ifdef DENSEPARTITION
    ,dense_partition_row_start(0),
    dense_partition_nrows(0),
    auto_set_dense_partition_percent(true),
    dense_partition_row_percent(0.0),
    nnz_persist_sptrimtx(0),
    nnz_rectspmtx(0),
    dense_trimtx(),
    row_map_rectspmtx(),
    entries_rectspmtx(),
    values_rectspmtx(),
    require_symbolic_numeric_dense_phase(false),
    numeric_complete(false)
#endif
  {
    this->set_if_algm_require_symb_lvlsched();
    this->set_if_algm_require_symb_chain();
#ifdef DENSEPARTITION
    this->set_if_algm_require_dense_partition();
#endif
  }

#if 0
  SPTRSVHandle ( SPTRSVAlgorithm choice, const size_type nrows_, bool lower_tri_, bool symbolic_complete_ = false ) :
    level_list( Kokkos::ViewAllocateWithoutInitializing("level_list"), nrows),
    nodes_per_level("nodes_per_level", nrows),
    nodes_grouped_by_level("nodes_grouped_by_level", nrows),
    nrows(nrows_),
    nlevel(0),
    lower_tri( lower_tri_ ),
    symbolic_complete( symbolic_complete_ ),
    algm(choice)
  {
    // WithoutInitializing
    Kokkos::deep_copy( level_list, signed_integral_t(-1) );
  }

/*
  template <class rhslno_row_view_t_,
            class rhslno_nnz_view_t_,
            class rhsscalar_nnz_view_t_,
            class rhsExecutionSpace,
            class rhsMemorySpace>
  SPTRSVHandle ( SPTRSVHandle< rhslno_row_view_t_, rhslno_nnz_view_t_, rhsscalar_nnz_view_t_, rhsExecutionSpace, rhsMemorySpace > & rhs ) {

    this->level_list = rhs.level_list;
    this->nodes_per_level = rhs.nodes_per_level;
    this->nodes_grouped_by_level = rhs.nodes_grouped_by_level;
    this->nrows = rhs.nrows;
    this->nlevel = rhs.nlevel;
    this->lower_tri = rhs.lower_tri;
    this->symbolic_complete = rhs.symbolic_complete;
    this->algm = rhs.algm;
  }

  template <class rhslno_row_view_t_,
            class rhslno_nnz_view_t_,
            class rhsscalar_nnz_view_t_,
            class rhsExecutionSpace,
            class rhsMemorySpace>
  SPTRSVHandle & operator= ( SPTRSVHandle< rhslno_row_view_t_, rhslno_nnz_view_t_, rhsscalar_nnz_view_t_, rhsExecutionSpace, rhsMemorySpace > & rhs ) {

    this->level_list = rhs.level_list;
    this->nodes_per_level = rhs.nodes_per_level;
    this->nodes_grouped_by_level = rhs.nodes_grouped_by_level;
    this->nrows = rhs.nrows;
    this->nlevel = rhs.nlevel;
    this->lower_tri = rhs.lower_tri;
    this->symbolic_complete = rhs.symbolic_complete;
    this->algm = rhs.algm;
    return *this;
  }
*/

#endif

  void init_handle(const size_type nrows_) {
    //set_nrows(nrows_);
    nrows = nrows_;

    // Assumed that level scheduling occurs during symbolic phase for all algorithms, for now
#ifdef DENSEPARTITION
    if ( this->require_symbolic_numeric_dense_phase == true ) {
      // This will set the dense_partition_nrows and dense_partition_row_start
      // From here, row start and nrows is known for each component - persist_sptrimtx, rect_spmtx, dense_trimtx
      std::cout << "  init_handle: dense allocs" << std::endl;

      if (auto_set_dense_partition_percent == true) {
        // The row start will depend on whether matrix is upper vs lower tri
        dense_partition_nrows = 0.01*nrows;
        std::cout << "  At handle init assume 1% dense_partition_nrows = " << dense_partition_nrows << std::endl;
      }
      else {
        // FIXME: The interface doesn't allow this to happen since the handle must be constructed before the percent can be set
        dense_partition_nrows = dense_partition_row_percent*nrows;
        std::cout << "  User provided dense_partition_nrows = " << dense_partition_nrows << std::endl;
      }

      dense_partition_row_start = lower_tri ? (nrows - dense_partition_nrows) : 0;

       // Lower
       // [0, dense_partition_row_start) sparse;  [dense_partition_row_start, nrows) dense, but map to [0,nrows-dense_partition_row_start)
       //
       // Upper
       // [dense_partition_row_start, nrows) sparse;  [0, dense_partition_nrows) dense, but map to [0,nrows-dense_partition_row_start)


      numeric_complete = false;
    }
    else {
      dense_trimtx= mtx_scalar_view_t();
      numeric_complete = false;
    }
#endif

#ifdef DENSEPARTITION
    if ( this->require_symbolic_lvlsched_phase == true && this->require_symbolic_numeric_dense_phase == false )
#else
    if ( this->require_symbolic_lvlsched_phase == true )
#endif
    {
      std::cout << "  init_handle: level schedule allocs" << std::endl;
      set_num_levels(0);
      level_list = signed_nnz_lno_view_t(Kokkos::ViewAllocateWithoutInitializing("level_list"), nrows_);
      Kokkos::deep_copy( level_list, signed_integral_t(-1) );
      nodes_per_level =  nnz_lno_view_t("nodes_per_level", nrows_);
      hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
      nodes_grouped_by_level = nnz_lno_view_t("nodes_grouped_by_level", nrows_);
      hnodes_grouped_by_level = Kokkos::create_mirror_view(nodes_grouped_by_level);
    }
#ifdef DENSEPARTITION
    else if ( this->require_symbolic_lvlsched_phase == true && this->require_symbolic_numeric_dense_phase == true)
    {
      // FIXME Do not use nrows (full matrix) in this case
      // Fixed - created a function to return correct value, this else-if routine can take over
      std::cout << "  init_handle: level schedule allocs" << std::endl;
      set_num_levels(0);
      level_list = signed_nnz_lno_view_t(Kokkos::ViewAllocateWithoutInitializing("level_list"), get_persist_sptrimtx_nrows() );
      Kokkos::deep_copy( level_list, signed_integral_t(-1) );
      nodes_per_level =  nnz_lno_view_t("nodes_per_level", get_persist_sptrimtx_nrows() );
      hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
      nodes_grouped_by_level = nnz_lno_view_t("nodes_grouped_by_level", get_persist_sptrimtx_nrows() );
      hnodes_grouped_by_level = Kokkos::create_mirror_view(nodes_grouped_by_level);
    }
#endif

    // TODO Incorporate usage of this data into the algorithms
    diagonal_offsets = nnz_lno_view_t(Kokkos::ViewAllocateWithoutInitializing("diagonal_offsets"), nrows_);

#ifdef DENSEPARTITION
    if (this->require_symbolic_chain_phase == true && this->require_symbolic_numeric_dense_phase == false )
#else
    if (this->require_symbolic_chain_phase == true)
#endif
    {
      std::cout << "  init_handle: chain ptr allocs" << std::endl;
      if (this->chain_threshold == -1) {
        // Need default if chain_threshold not set
        // 0: Every level, regardless of number of nodes, is launched within a kernel
        if (team_size == -1) {
          this->chain_threshold = 0; 
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", this->nrows);
        }
        else {
          std::cout << "  Warning: chain_threshold was not set - will default to team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          this->chain_threshold = this->team_size; 
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", this->nrows);
        }
      }
      else {
        // FIXME Compare threshold with team_size limit - either error or automatically adjust if incompatible
        if (this->team_size >= this->chain_threshold) {
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", this->nrows);
        }
        else if (this->team_size == -1) {
          std::cout << "  Warning: team_size was not set  team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          std::cout << "  Automatically setting team_size to chain_threshold - if this exceeds the hardware limitation a runtime error will occur during kernel launch - reduce chain_threshold in that case" << std::endl;
          this->team_size = this->chain_threshold;
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", this->nrows);
        }
        else {
          // TODO Must set team_size when using chain - or should it be automatically set to chain_threshold?
          std::cout << "  EXPERIMENTAL: team_size < chain_size: team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", this->nrows);
          //std::cout << "  Error: team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          //throw std::runtime_error ("  sptrsv_handle.init_handle error: chain_threshold > team_size - this is an invalid pair of values for this algorithm");
        }
      }
    }
#ifdef DENSEPARTITION
    else if (this->require_symbolic_chain_phase == true && this->require_symbolic_numeric_dense_phase == true )
    {
      // Fixed - created a function to return correct value, this else-if routine can take over
      std::cout << "  init_handle: chain ptr allocs" << std::endl;
      if (this->chain_threshold == -1) {
        // Need default if chain_threshold not set
        // 0: Every level, regardless of number of nodes, is launched within a kernel
        if (team_size == -1) {
          this->chain_threshold = 0; 
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_persist_sptrimtx_nrows() );
        }
        else {
          std::cout << "  Warning: chain_threshold was not set - will default to team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          this->chain_threshold = this->team_size; 
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_persist_sptrimtx_nrows() );
        }
      }
      else {
        // FIXME Compare threshold with team_size limit - either error or automatically adjust if incompatible
        if (this->team_size >= this->chain_threshold) {
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_persist_sptrimtx_nrows() );
        }
        else if (this->team_size == -1) {
          std::cout << "  Warning: team_size was not set  team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          std::cout << "  Automatically setting team_size to chain_threshold - if this exceeds the hardware limitation a runtime error will occur during kernel launch - reduce chain_threshold in that case" << std::endl;
          this->team_size = this->chain_threshold;
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_persist_sptrimtx_nrows() );
        }
        else {
          // TODO Must set team_size when using chain - or should it be automatically set to chain_threshold?
          std::cout << "  EXPERIMENTAL: team_size < chain_size: team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_persist_sptrimtx_nrows() );
          //std::cout << "  Error: team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          //throw std::runtime_error ("  sptrsv_handle.init_handle error: chain_threshold > team_size - this is an invalid pair of values for this algorithm");
        }
      }
    }
#endif
    else {
      h_chain_ptr = host_signed_nnz_lno_view_t();
      this->chain_threshold = -1;
    }
    set_num_chain_entries(0);
    set_symbolic_incomplete();
  }


  virtual ~SPTRSVHandle() {};

  bool algm_requires_symb_lvlsched() const { return require_symbolic_lvlsched_phase; } 

  bool algm_requires_symb_chain() const { return require_symbolic_chain_phase; }

  // TODO set_algorithm should reset the handle depending on which algms are being switched...
  // i.e. "Compatible algorithms"
  void set_algorithm(SPTRSVAlgorithm choice) { 
    algm = choice; 
  }

  void reset_algorithm(SPTRSVAlgorithm choice) { 
    if (algm != choice) {
      algm = choice; 
      init_handle(nrows);
    }
  }

  KOKKOS_INLINE_FUNCTION
  SPTRSVAlgorithm get_algorithm() { return algm; }

  KOKKOS_INLINE_FUNCTION
  signed_nnz_lno_view_t get_level_list() const { return level_list; }

  inline
  host_signed_nnz_lno_view_t get_host_level_list() const { 
    auto hlevel_list = Kokkos::create_mirror_view(this->level_list);
    Kokkos::deep_copy(hlevel_list, this->level_list);
    return hlevel_list; 
  }

  KOKKOS_INLINE_FUNCTION
  nnz_lno_view_t get_diagonal_offsets() const { return diagonal_offsets; }

  inline
  host_signed_nnz_lno_view_t get_host_chain_ptr() const { return h_chain_ptr; }

  KOKKOS_INLINE_FUNCTION
  nnz_lno_view_t get_nodes_per_level() const { return nodes_per_level; }

  inline
  host_nnz_lno_view_t get_host_nodes_per_level() const { 
//    auto hnodes_per_level = Kokkos::create_mirror_view(this->nodes_per_level);
//    Kokkos::deep_copy(hnodes_per_level, this->nodes_per_level);
    return hnodes_per_level; 
  }

  KOKKOS_INLINE_FUNCTION
  nnz_lno_view_t get_nodes_grouped_by_level() const { return nodes_grouped_by_level; }

  inline
  host_nnz_lno_view_t get_host_nodes_grouped_by_level() const { return hnodes_grouped_by_level; }

  KOKKOS_INLINE_FUNCTION
  size_type get_nrows() const { return nrows; }
  void set_nrows(const size_type nrows_) { this->nrows = nrows_; }

#ifdef DENSEPARTITION
  // TODO FIXME May need to add a create_set routine, first will try allocating and initializing within symbolic
  KOKKOS_INLINE_FUNCTION
  nnz_row_view_t get_row_map_rectspmtx() { return row_map_rectspmtx; }

  inline
  void alloc_row_map_rectspmtx(size_type dim) { row_map_rectspmtx = nnz_row_view_t("row_map_rectspmtx", dim); }

  KOKKOS_INLINE_FUNCTION
  nnz_lno_view_t get_entries_rectspmtx() { return entries_rectspmtx; }

  inline
  void alloc_entries_rectspmtx(size_type dim) { entries_rectspmtx = nnz_lno_view_t("entries_rectspmtx", dim); }

  KOKKOS_INLINE_FUNCTION
  nnz_scalar_view_t get_values_rectspmtx() { return values_rectspmtx; }

  void alloc_values_rectspmtx(size_type dim) { values_rectspmtx = nnz_scalar_view_t("values_rectspmtx", dim); }

  inline
  void set_dense_partition_row_percent(const float rp) { 
    if (numeric_complete == false && symbolic_complete == false) {
      dense_partition_row_percent = rp;
      std::cout << "    manual set dense_partition_row_percent = " << dense_partition_row_percent << std::endl;
      auto_set_dense_partition_percent = false;
      dense_partition_nrows = dense_partition_row_percent*nrows;
      std::cout << "  nrows = " << nrows << std::endl;
      std::cout << "  User resultant dense_partition_nrows = " << dense_partition_nrows << std::endl;
      dense_partition_row_start = lower_tri ? (nrows - dense_partition_nrows) : 0;
      std::cout << "  User resultant dense_row_start = " << dense_partition_row_start << std::endl;
    }
    else {
      std::cout << "  dprs can not be set after symbolic and numeric phase, request ignored - destroy and create a new handle to do this for now." << std::endl;
    }
  }

  KOKKOS_INLINE_FUNCTION
  size_type get_dense_partition_row_start() const { return dense_partition_row_start; }

  KOKKOS_INLINE_FUNCTION
  size_type get_dense_partition_nrows() const { return dense_partition_nrows; }


  KOKKOS_INLINE_FUNCTION
  size_type get_rectspmtx_row_start() const { return dense_partition_row_start; }

  KOKKOS_INLINE_FUNCTION
  size_type get_rectspmtx_nrows() const { return dense_partition_nrows; }

  KOKKOS_INLINE_FUNCTION
  size_type get_rectspmtx_ncols() const { return (nrows - dense_partition_nrows); }

  KOKKOS_INLINE_FUNCTION
  size_type get_nnz_rectspmtx() const { return nnz_rectspmtx; }

  inline
  void set_nnz_rectspmtx(const size_type snnz) { nnz_rectspmtx = snnz; }


  //KOKKOS_INLINE_FUNCTION
  //size_type get_persist_sptrimtx_row_start() const { return lower_tri ? 0 : dense_partition_nrows; }

  KOKKOS_INLINE_FUNCTION
  size_type get_persist_sptrimtx_row_start() const { return lower_tri ? size_type(0) : dense_partition_nrows; }

  KOKKOS_INLINE_FUNCTION
  size_type get_persist_sptrimtx_nrows() const { return require_symbolic_numeric_dense_phase ? (nrows - dense_partition_nrows) : nrows; }

  // Needed for the solve phase
  KOKKOS_INLINE_FUNCTION
  size_type get_nnz_persist_sptrimtx() const { return nnz_persist_sptrimtx; }

  // Set in symbolic...
  inline
  void set_nnz_persist_sptrimtx(const size_type snnz) { nnz_persist_sptrimtx = snnz; }


  // TODO This is determined by upper vs lower tri and percent of dense rows - handle input params; this should not change unless matrix is changed, requiring reset of the handle
/*
  inline
  void set_dense_partition_row_start(const size_type sr) { 
    dense_partition_row_start = sr;
    std::cout << "    manual set dense_partition_row_start = " << dense_partition_row_start << std::endl;
  }
*/

/*
  // TODO Have this set internally based on inputs, not mutable by user
  inline
  void set_dense_partition_nrows(const size_type nr) { 
    dense_partition_nrows = nr;
    std::cout << "    manual set dense_partition_nrows = " << dense_partition_nrows << std::endl;
    //auto_set_dense_partition_row_end = false;
  }
*/



  KOKKOS_INLINE_FUNCTION
  mtx_scalar_view_t get_dense_trimtx() { return dense_trimtx; }

  inline
  void alloc_dense_trimtx(size_type dim0, size_type dim1) { dense_trimtx = mtx_scalar_view_t("dense_trimtx", dim0, dim1); }

// FIXME Unused - remove, if they turn out to be incorrect/unnecessary
/*
  // For a single "dense" block, the nnz for the persisting sparse partition is also the offset into the start of the dense entries (indices) when using the original entries array
  // TODO If recursion or multiple dense "stripes" are allowed, then this routine will no longer hold and will need to return value from an array storing the offset start per stripe
  KOKKOS_INLINE_FUNCTION
  size_type get_dense_entries_start_offset() const { return nnz_persist_sptrimtx; }

  // FIXME - the variable name is not consistent
  // This may no longer be necessary...
  inline
  void set_dense_entries_start_offset(const size_type snnz) { nnz_persist_sptrimtx = snnz; }
*/



  // TODO
  // Add routines to return start/end of level schedule symbolic routines
  // FIXME Change this to allow for partitioned sparse mtx
  KOKKOS_INLINE_FUNCTION
  size_type get_lvlsched_node_start() {
    if (lower_tri) {
      // Non-partitioned: 0; 
      //if (require_symbolic_numeric_dense_phase == true)
      return require_symbolic_numeric_dense_phase == true ? 0 : 0;
    }
    else {
      // Non-partitioned: nrows-1
      // Upper tri - "start" level scheduling from bottom-right corner of the matrix
      return require_symbolic_numeric_dense_phase == true ? nrows-1 : nrows-1;
    }
  }

  KOKKOS_INLINE_FUNCTION
  size_type get_lvlsched_node_end() {
    if (lower_tri) {
      // Non-partitioned: nrows-1
      return require_symbolic_numeric_dense_phase == true ? (nrows - dense_partition_nrows) : nrows;
    }
    else {
      // Non-partitioned: 0
      return require_symbolic_numeric_dense_phase == true ? dense_partition_nrows : 0;
    }
  }
  
#endif

  // FIXME This is only interface for setting the chain_threshold for now, but results in unnecessary realloc of h_chain_ptr
  void reset_chain_threshold(const signed_integral_t threshold) {
    // TODO Must check that team_size corresponding to chain_threshold is valid, but requires instantiating the kernel to get max_team_size
    // NOTE: Below span() is being used as a proxy for an uninitialized h_chain_ptr (i.e. 0 length)
    if (threshold != this->chain_threshold || h_chain_ptr.span() == 0) {
        this->chain_threshold = threshold;
        if (this->team_size >= this->chain_threshold) {
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", this->nrows);
        }
        else if (this->team_size == -1) {
          std::cout << "  Warning: team_size was not set  team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          std::cout << "  Automatically setting team_size to chain_threshold - if this exceeds the hardware limitation a runtime error will occur during kernel launch - reduce chain_threshold in that case" << std::endl;
          this->team_size = this->chain_threshold;
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", this->nrows);
        }
        else {
          // TODO Must set team_size when using chain - or should it be automatically set to chain_threshold?
          std::cout << "  EXPERIMENTAL: team_size < chain_size: team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", this->nrows);
          //std::cout << "  Error: team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          //throw std::runtime_error ("  sptrsv_handle.init_handle error: chain_threshold > team_size - this is an invalid pair of values for this algorithm");
        }
    }
  }

  KOKKOS_INLINE_FUNCTION
  signed_integral_t get_chain_threshold () const { return this->chain_threshold; }


  bool is_lower_tri() const { return lower_tri; }
  bool is_upper_tri() const { return !lower_tri; }

  bool is_symbolic_complete() const { return symbolic_complete; }

  KOKKOS_INLINE_FUNCTION
  size_type get_num_levels() const { return nlevel; }

  void set_num_levels(size_type nlevels_) { this->nlevel = nlevels_; }

  void set_symbolic_complete() { this->symbolic_complete = true; }
  void set_symbolic_incomplete() { this->symbolic_complete = false; }

#ifdef DENSEPARTITION
  bool is_numeric_complete() const { return numeric_complete; }

  void set_numeric_complete() { this->numeric_complete = true; }
  void set_numeric_incomplete() { this->numeric_complete = false; }
#endif

  KOKKOS_INLINE_FUNCTION
  int get_team_size() const {return this->team_size;}
  void set_team_size(const int ts) {this->team_size = ts;}

  KOKKOS_INLINE_FUNCTION
  int get_vector_size() const {return this->vector_size;}
  void set_vector_size(const int vs) {this->vector_size = vs;}

  KOKKOS_INLINE_FUNCTION
  int get_num_chain_entries() const {return this->num_chain_entries;}
  void set_num_chain_entries(const int nce) {this->num_chain_entries = nce;}

  void print_algorithm() { 
    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_RP )
      std::cout << "SEQLVLSCHD_RP" << std::endl;;

    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_TP1 )
      std::cout << "SEQLVLSCHD_TP1" << std::endl;;

    if ( algm == SPTRSVAlgorithm::SEQLVLSCHED_TP2 ) {
      std::cout << "SEQLVLSCHED_TP2" << std::endl;;
      std::cout << "WARNING: With CUDA this is currently only reliable with int-int ordinal-offset pair" << std::endl;
    }

    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_TP1CHAIN )
      std::cout << "SEQLVLSCHD_TP1CHAIN" << std::endl;;

#ifdef DENSEPARTITION
    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1 )
      std::cout << "SEQLVLSCHD_DENSEP_TP1" << std::endl;;
#endif
  }


  std::string return_algorithm_string() { 
    std::string ret_string;

    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_RP )
      ret_string = "SEQLVLSCHD_RP";

    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_TP1 )
      ret_string = "SEQLVLSCHD_TP1";

    if ( algm == SPTRSVAlgorithm::SEQLVLSCHED_TP2 )
      ret_string = "SEQLVLSCHED_TP2";

    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_TP1CHAIN )
      ret_string = "SEQLVLSCHD_TP1CHAIN";

#ifdef DENSEPARTITION
    if ( algm == SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1 )
      ret_string = "SEQLVLSCHD_DENSEP_TP1";
#endif

    return ret_string;
  }


  inline SPTRSVAlgorithm StringToSPTRSVAlgorithm(std::string & name) {
    if(name=="SPTRSV_DEFAULT")                return SPTRSVAlgorithm::SEQLVLSCHD_RP;
    else if(name=="SPTRSV_RANGEPOLICY")       return SPTRSVAlgorithm::SEQLVLSCHD_RP;
    else if(name=="SPTRSV_TEAMPOLICY1")       return SPTRSVAlgorithm::SEQLVLSCHD_TP1;
    else if(name=="SPTRSV_TEAMPOLICY2")       return SPTRSVAlgorithm::SEQLVLSCHED_TP2;
    else if(name=="SPTRSV_TEAMPOLICY1CHAIN")  return SPTRSVAlgorithm::SEQLVLSCHD_TP1CHAIN;
#ifdef DENSEPARTITION
    else if(name=="SPTRSV_DENSEPARTITION_TEAMPOLICY1") return  SPTRSVAlgorithm::SEQLVLSCHD_DENSEP_TP1;
#endif
    else
      throw std::runtime_error("Invalid SPTRSVAlgorithm name");
  }

};

} // namespace Experimental
} // namespace Kokkos

#endif
