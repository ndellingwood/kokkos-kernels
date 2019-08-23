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

  typedef typename std::remove_const<scalar_t_>::type  nnz_scalar_t;
  typedef const nnz_scalar_t const_nnz_scalar_t;


  typedef typename Kokkos::View<size_type *, HandleTempMemorySpace> nnz_row_view_temp_t;
  typedef typename Kokkos::View<size_type *, HandlePersistentMemorySpace> nnz_row_view_t;
  typedef typename nnz_row_view_t::HostMirror host_nnz_row_view_t;
 // typedef typename row_lno_persistent_work_view_t::HostMirror row_lno_persistent_work_host_view_t; //Host view type

  typedef typename Kokkos::View<nnz_scalar_t *, HandleTempMemorySpace> nnz_scalar_view_temp_t;
  typedef typename Kokkos::View<nnz_scalar_t *, HandlePersistentMemorySpace> nnz_scalar_view_t;
  typedef typename nnz_scalar_view_t::HostMirror host_nnz_scalar_view_t;


  typedef typename Kokkos::View<nnz_lno_t *, HandleTempMemorySpace> nnz_lno_view_temp_t;
  typedef typename Kokkos::View<nnz_lno_t *, HandlePersistentMemorySpace> nnz_lno_view_t;
  typedef typename nnz_lno_view_t::HostMirror host_nnz_lno_view_t;
 // typedef typename nnz_lno_persistent_work_view_t::HostMirror nnz_lno_persistent_work_host_view_t; //Host view type


  typedef typename std::make_signed<typename nnz_row_view_t::non_const_value_type>::type signed_integral_t;
  typedef Kokkos::View< signed_integral_t*, typename nnz_row_view_t::array_layout, typename nnz_row_view_t::device_type, typename nnz_row_view_t::memory_traits > signed_nnz_lno_view_t;
  typedef typename signed_nnz_lno_view_t::HostMirror host_signed_nnz_lno_view_t;


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
  // KokkosBlas::gemv expects rank2 view for the matrix
  typedef typename Kokkos::View<nnz_scalar_t **, HandlePersistentMemorySpace> mtx_scalar_view_t;
  size_type dense_start_row;
  size_type num_sparse_part_nnz;

  mtx_scalar_view_t dense_matrix_partition;
  mtx_scalar_view_t dense_tri_partition; // 2D storage is inefficient for this matrix, but fits cuBLAS pattern...
  //mtx_scalar_view_t dense_tri_inv_partition; // 2D storage is inefficient for this matrix, but fits cuBLAS pattern...
  //nnz_scalar_view_t dense_bvector_partition; // Make this a copy of subview of the original view - TODO Unneccessary to add here, just create subview before fcn call?

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

  KOKKOS_INLINE_FUNCTION
  size_type get_sparse_num_rows() { return require_symbolic_numeric_dense_phase ? dense_start_row : nrows; }
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
    ,dense_start_row(0),
    num_sparse_part_nnz(0),
    dense_matrix_partition(),
    dense_tri_partition(),
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
    set_nrows(nrows_);
    // Assumed that level scheduling occurs during symbolic phase for all algorithms, for now

#ifdef DENSEPARTITION
    // FIXME This algorithm still uses lvl scheduling, but nrows refers to the full matrix, and we now need the lvl scheduling to use num rows of the sparse partition
    if ( this->require_symbolic_numeric_dense_phase == true ) {
      dense_start_row = 0.75*nrows; // TODO Set this differently, just simple default for now
      auto num_dense_rows = nrows - dense_start_row;
       // [0, dense_start_row) sparse;  [dense_start_row, nrows) dense, but map to [0,nrows-dense_start_row)

      dense_matrix_partition = mtx_scalar_view_t("dense_mtx",  num_dense_rows, dense_start_row);
      dense_tri_partition = mtx_scalar_view_t("dense_tri_mtx", num_dense_rows, num_dense_rows);
      //dense_tri_inv_partition = mtx_scalar_view_t ; // 2D storage is inefficient for this matrix, but fits cuBLAS pattern...

      numeric_complete = false;
    }
#endif

#ifdef DENSEPARTITION
    if ( this->require_symbolic_lvlsched_phase == true && this->require_symbolic_numeric_dense_phase == false )
#else
    if ( this->require_symbolic_lvlsched_phase == true )
#endif
    {
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
      set_num_levels(0);
      level_list = signed_nnz_lno_view_t(Kokkos::ViewAllocateWithoutInitializing("level_list"), get_sparse_num_rows() );
      Kokkos::deep_copy( level_list, signed_integral_t(-1) );
      nodes_per_level =  nnz_lno_view_t("nodes_per_level", get_sparse_num_rows() );
      hnodes_per_level = Kokkos::create_mirror_view(nodes_per_level);
      nodes_grouped_by_level = nnz_lno_view_t("nodes_grouped_by_level", get_sparse_num_rows() );
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
      if (this->chain_threshold == -1) {
        // Need default if chain_threshold not set
        // 0: Every level, regardless of number of nodes, is launched within a kernel
        if (team_size == -1) {
          this->chain_threshold = 0; 
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_sparse_num_rows() );
        }
        else {
          std::cout << "  Warning: chain_threshold was not set - will default to team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          this->chain_threshold = this->team_size; 
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_sparse_num_rows() );
        }
      }
      else {
        // FIXME Compare threshold with team_size limit - either error or automatically adjust if incompatible
        if (this->team_size >= this->chain_threshold) {
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_sparse_num_rows() );
        }
        else if (this->team_size == -1) {
          std::cout << "  Warning: team_size was not set  team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          std::cout << "  Automatically setting team_size to chain_threshold - if this exceeds the hardware limitation a runtime error will occur during kernel launch - reduce chain_threshold in that case" << std::endl;
          this->team_size = this->chain_threshold;
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_sparse_num_rows() );
        }
        else {
          // TODO Must set team_size when using chain - or should it be automatically set to chain_threshold?
          std::cout << "  EXPERIMENTAL: team_size < chain_size: team_size = " << this->team_size << "  chain_threshold = " << this->chain_threshold << std::endl;
          h_chain_ptr = host_signed_nnz_lno_view_t("h_chain_ptr", get_sparse_num_rows() );
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
  KOKKOS_INLINE_FUNCTION
  size_type get_dense_start_row() const { return dense_start_row; }
/*
  // Below - these were intended for the dense matrix, but naming is unclear; this revealed that I should just use the View extents to get these values
  // Leaving for now as reminder
  KOKKOS_INLINE_FUNCTION
  size_type get_num_dense_rows() const { return (nrows - dense_start_row); }

  KOKKOS_INLINE_FUNCTION
  size_type get_num_dense_cols() const { return (dense_start_row); }
*/

  KOKKOS_INLINE_FUNCTION
  size_type get_num_sparse_part_nnz() const { return num_sparse_part_nnz; }

  inline
  void set_num_sparse_part_nnz(const size_type snnz) { num_sparse_part_nnz = snnz; }

  KOKKOS_INLINE_FUNCTION
  mtx_scalar_view_t get_dense_mtx_partition() const { return dense_matrix_partition; }

  KOKKOS_INLINE_FUNCTION
  mtx_scalar_view_t get_dense_tri_partition() const { return dense_tri_partition; }
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
