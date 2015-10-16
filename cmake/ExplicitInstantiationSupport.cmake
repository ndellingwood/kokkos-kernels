MESSAGE(STATUS "${PACKAGE_NAME}: Processing ETI / test support")

# This CMake module generates the following header file, which gets
# written to the build directory (like other header files that CMake
# generates).  The file contains macros that do instantiation over a
# finite set of template parameter combinations.  We use the macros
# both for ETI (explicit template instantiation), and for tests.
# Thus, the macros need to be generated even if ETI is OFF.
SET(${PACKAGE_NAME}_ETI_FILE ${PACKAGE_NAME}_ETIHelperMacros.h)
SET(${PACKAGE_NAME}_ETI_FILE_PATH ${${PACKAGE_NAME}_BINARY_DIR}/src/${${PACKAGE_NAME}_ETI_FILE})

#
# Users have the option to generate the above header file themselves.
# We prefer that users let Trilinos generate the header file.
# However, folks who make intense use of TriBITS sometimes find that
# reusing a previously generated header file shaves a couple minutes
# off their CMake configuration time.  Thus, we give them that option.
#

ADVANCED_SET(${PACKAGE_NAME}_USE_STATIC_ETI_MACROS_HEADER_FILE ""
  CACHE PATH
  "If set, gives the path to a static version of the file ${${PACKAGE_NAME}_ETI_FILE}.  If not set (the default), then the file is generated automatically."
  )

IF(${PACKAGE_NAME}_USE_STATIC_ETI_MACROS_HEADER_FILE)
  # The user wants us to accept their header file and not generate one.
  MESSAGE("-- NOTE: Skipping generation and using provided static file"
     " '${${PACKAGE_NAME}_USE_STATIC_ETI_MACROS_HEADER_FILE}'")
  CONFIGURE_FILE(
    ${${PACKAGE_NAME}_USE_STATIC_ETI_MACROS_HEADER_FILE}
    ${${PACKAGE_NAME}_ETI_FILE_PATH}
    COPYONY
    )
  RETURN()
ENDIF()

#
# The user wants us to generate the header file.  This is the default
# behavior.
#

# Tpetra ETI type fields.  S, LO, EX correspond to the template
# parameters Scalar, LocalOrdinal, and ExecutionSpace.  TpetraKernels
# does not need to know about GlobalOrdinal so we omit that.
SET(${PACKAGE_NAME}_ETI_FIELDS "S|LO|EX")

# Set up a pattern that excludes all Scalar types that are also
# possible GlobalOrdinal types, but includes all other types.
# TriBITS' ETI system knows how to interpret this pattern.
#
# FIXME (mfh 17 Aug 2015, 16 Oct 2015) A better way to do this would
# be to subtract away all enabled GlobalOrdinal types.  Plus, what if
# someone really wants a CrsMatrix<int,...>?
TRIBITS_ETI_TYPE_EXPANSION(${PACKAGE_NAME}_ETI_EXCLUDE_SET_ORDINAL_SCALAR "S=short|short int|unsigned short|unsigned short int|int|unsigned|unsigned int|long|long int|unsigned long|unsigned long int|long long|long long int|unsigned long long|unsigned long long int|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t|size_t|ptrdiff_t" "LO=.*" "EX=.*")

# TriBITS' ETI system expects a set of types to be a string, delimited
# by |.  Each template parameter (e.g., Scalar, LocalOrdinal, ...) has
# its own set.  The JOIN commands below set up those lists.  We use
# the following sets that this subpackage defines:
#
# Scalar: ${PACKAGE_NAME}_ETI_SCALARS
# LocalOrdinal: ${PACKAGE_NAME}_ETI_LORDS
# ExecutionSpace: ${PACKAGE_NAME}_ETI_EXECUTION_SPACES
#
# Note that the Scalar set from Tpetra includes the Scalar =
# GlobalOrdinal case.  We have to exclude that explicitly in what
# follows.
JOIN(${PACKAGE_NAME}_ETI_SCALARS "|" FALSE ${${PACKAGE_NAME}_ETI_SCALARS})
JOIN(${PACKAGE_NAME}_ETI_LORDS "|" FALSE ${${PACKAGE_NAME}_ETI_LORDS})
JOIN(${PACKAGE_NAME}_ETI_EXECUTION_SPACES "|" FALSE ${${PACKAGE_NAME}_ETI_EXECUTION_SPACES})

MESSAGE(STATUS "Enabled Scalar types:         ${${PACKAGE_NAME}_ETI_SCALARS}")
MESSAGE(STATUS "Enabled LocalOrdinal types:   ${${PACKAGE_NAME}_ETI_LORDS}")
MESSAGE(STATUS "Enabled ExecutionSpace types: ${${PACKAGE_NAME}_ETI_EXECUTION_SPACES}")

# Construct the "type expansion" string that TriBITS' ETI system
# expects.  Even if ETI is OFF, we will use this to generate macros
# for instantiating tests.
TRIBITS_ETI_TYPE_EXPANSION(SingleScalarInsts 
  "S=${${PACKAGE_NAME}_ETI_SCALARS}"
  "LO=${${PACKAGE_NAME}_ETI_LORDS}" 
  "EX=${${PACKAGE_NAME}_ETI_EXECUTION_SPACES}")

# Set up the set of enabled type combinations, in a format that
# TriBITS understands.
#
# mfh 17 Aug 2015, 16 Oct 2015: I don't exactly understand what's
# going on here, but it looks like if ETI is enabled, we let users
# modify ${PACKAGE_NAME}_ETI_LIBRARYSET, and if it's not, we don't.
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)
IF(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)
  TRIBITS_ADD_ETI_INSTANTIATIONS(${PACKAGE_NAME} ${SingleScalarInsts})
  # mfh 17 Aug 2015: For some reason, these are empty unless ETI is
  # ON.  That seems to be OK, though, unless users want to exclude
  # enabling certain type combinations when ETI is OFF.
  MESSAGE(STATUS "Excluded type combinations: ${${PACKAGE_NAME}_ETI_EXCLUDE_SET}:${${PACKAGE_NAME}_ETI_EXCLUDE_SET_INT}")
ELSE()
  TRIBITS_ETI_TYPE_EXPANSION(${PACKAGE_NAME}_ETI_LIBRARYSET 
    "S=${${PACKAGE_NAME}_ETI_SCALARS}"
    "LO=${${PACKAGE_NAME}_ETI_LORDS}" 
    "EX=${${PACKAGE_NAME}_ETI_EXECUTION_SPACES}")
ENDIF()
MESSAGE(STATUS "Set of enabled types, before exclusions: ${${PACKAGE_NAME}_ETI_LIBRARYSET}")

#
# Generate the instantiation macros.  These go into
# ${PACKAGE_NAME}_ETIHelperMacros.h, which is generated from
# ${PACKAGE_NAME}_ETIHelperMacros.h.in (in this directory).
#

# Generate macros that exclude possible GlobalOrdinal types from the
# list of Scalar types.
TRIBITS_ETI_GENERATE_MACROS(
    "${${PACKAGE_NAME}_ETI_FIELDS}" "${${PACKAGE_NAME}_ETI_LIBRARYSET}" 
    "${${PACKAGE_NAME}_ETI_EXCLUDE_SET};${${PACKAGE_NAME}_ETI_EXCLUDE_SET_ORDINAL_SCALAR}"  
    list_of_manglings eti_typedefs
    "TPETRAKERNELS_INSTANTIATE_SLX_NO_ORDINAL_SCALAR(S,LO,EX)"  TPETRAKERNELS_INSTANTIATE_SLX_NO_ORDINAL_SCALAR
    "TPETRAKERNELS_INSTANTIATE_SL_NO_ORDINAL_SCALAR(S,LO)"     TPETRAKERNELS_INSTANTIATE_SL_NO_ORDINAL_SCALAR
    "TPETRAKERNELS_INSTANTIATE_SX_NO_ORDINAL_SCALAR(S,EX)"      TPETRAKERNELS_INSTANTIATE_SX_NO_ORDINAL_SCALAR
    "TPETRAKERNELS_INSTANTIATE_S_NO_ORDINAL_SCALAR(S)"         TPETRAKERNELS_INSTANTIATE_S_NO_ORDINAL_SCALAR)

# Generate macros that include all Scalar types (if applicable),
# including possible GlobalOrdinal types.
TRIBITS_ETI_GENERATE_MACROS(
    "${${PACKAGE_NAME}_ETI_FIELDS}" "${${PACKAGE_NAME}_ETI_LIBRARYSET}" 
    "${${PACKAGE_NAME}_ETI_EXCLUDE_SET};${${PACKAGE_NAME}_ETI_EXCLUDE_SET_INT}"  
    list_of_manglings eti_typedefs
    "TPETRAKERNELS_INSTANTIATE_SLX(S,LO,EX)"  TPETRAKERNELS_INSTANTIATE_SLX
    "TPETRAKERNELS_INSTANTIATE_SL(S,LO)"      TPETRAKERNELS_INSTANTIATE_SL
    "TPETRAKERNELS_INSTANTIATE_SN(S,EX)"      TPETRAKERNELS_INSTANTIATE_SX
    "TPETRAKERNELS_INSTANTIATE_S(S)"          TPETRAKERNELS_INSTANTIATE_S
    "TPETRAKERNELS_INSTANTIATE_LX(LO,EX)"     TPETRAKERNELS_INSTANTIATE_LX
    "TPETRAKERNELS_INSTANTIATE_L(LO)"         TPETRAKERNELS_INSTANTIATE_L
    "TPETRAKERNELS_INSTANTIATE_X(EX)"         TPETRAKERNELS_INSTANTIATE_X)

# Generate "mangled" typedefs.  Macros sometimes get grumpy when types
# have spaces, colons, or angle brackets in them.  This includes types
# like "long long" or "std::complex<double>".  Thus, we define
# typedefs that remove the offending characters.  The typedefs also
# get written to the generated header file.
TRIBITS_ETI_GENERATE_TYPEDEF_MACRO(TPETRAKERNELS_ETI_TYPEDEFS "TPETRAKERNELS_ETI_MANGLING_TYPEDEFS" "${eti_typedefs}")

# Generate the header file ${PACKAGE_NAME}_ETIHelperMacros.h, from the
# file ${PACKAGE_NAME}_ETIHelperMacros.h.in (that lives in this
# directory).  The generated header file gets written to the Trilinos
# build directory, in packages/tpetra/kernels/src/.
CONFIGURE_FILE(
  ${${PACKAGE_NAME}_SOURCE_DIR}/cmake/${PACKAGE_NAME}_ETIHelperMacros.h.in
  ${${PACKAGE_NAME}_ETI_FILE_PATH}
  )