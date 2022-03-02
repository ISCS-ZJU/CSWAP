#include "zfp.h"
#include "types.h"
#include "system.h"
#include "version.h"
#include "bitstream.h"
#include "macros.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>


#define wsize ((uint)(CHAR_BIT * sizeof(uint64)))
const size_t stream_word_bits = wsize;

size_t
zfp_type_size(zfp_type type)
{
  switch (type) {
    case zfp_type_int32:
      return sizeof(int32);
    case zfp_type_int64:
      return sizeof(int64);
    case zfp_type_float:
      return sizeof(float);
    case zfp_type_double:
      return sizeof(double);
    default:
      return 0;
  }
}

zfp_stream*
zfp_stream_open(bitstream* stream)
{
  zfp_stream* zfp = (zfp_stream*)malloc(sizeof(zfp_stream));
  if (zfp) {
    zfp->stream = stream;
    zfp->minbits = ZFP_MIN_BITS;
    zfp->maxbits = ZFP_MAX_BITS;
    zfp->maxprec = ZFP_MAX_PREC;
    zfp->minexp = ZFP_MIN_EXP;
    zfp->exec.policy = zfp_exec_serial;
  }
  return zfp;
}


void
zfp_stream_close(zfp_stream* zfp)
{
  free(zfp);
}

static zfp_bool
is_reversible(const zfp_stream* zfp)
{
  return zfp->minexp < ZFP_MIN_EXP;
}

uint
zfp_field_dimensionality(const zfp_field* field)
{
  return field->nx ? field->ny ? field->nz ? field->nw ? 4 : 3 : 2 : 1 : 0;
}

uint
zfp_field_precision(const zfp_field* field)
{
  return (uint)(CHAR_BIT * zfp_type_size(field->type));
}

size_t
zfp_field_size(const zfp_field* field, size_t* size)
{
  if (size)
    switch (zfp_field_dimensionality(field)) {
      case 4:
        size[3] = field->nw;
        /* FALLTHROUGH */
      case 3:
        size[2] = field->nz;
        /* FALLTHROUGH */
      case 2:
        size[1] = field->ny;
        /* FALLTHROUGH */
      case 1:
        size[0] = field->nx;
        break;
    }
  return MAX(field->nx, 1u) * MAX(field->ny, 1u) * MAX(field->nz, 1u) * MAX(field->nw, 1u);
}


size_t
zfp_stream_maximum_size(const zfp_stream* zfp, const zfp_field* field)
{
  zfp_bool reversible = is_reversible(zfp);
  uint dims = zfp_field_dimensionality(field);
  //printf("dims:%d",dims);
  size_t mx = (MAX(field->nx, 1u) + 3) / 4;
  size_t my = (MAX(field->ny, 1u) + 3) / 4;
  size_t mz = (MAX(field->nz, 1u) + 3) / 4;
  size_t mw = (MAX(field->nw, 1u) + 3) / 4;
  size_t blocks = mx * my * mz * mw;
  uint values = 1u << (2 * dims);
  uint maxbits = 0;

  if (!dims)
    return 0;
  switch (field->type) {
    case zfp_type_int32:
      maxbits += reversible ? 5 : 0;
      break;
    case zfp_type_int64:
      maxbits += reversible ? 6 : 0;
      break;
    case zfp_type_float:
      maxbits += reversible ? 1 + 1 + 8 + 5 : 1 + 8;
      break;
    case zfp_type_double:
      maxbits += reversible ? 1 + 1 + 11 + 6 : 1 + 11;
      break;
    default:
      return 0;
  }
  maxbits += values - 1 + values * MIN(zfp->maxprec, zfp_field_precision(field));
  maxbits = MIN(maxbits, zfp->maxbits);
  maxbits = MAX(maxbits, zfp->minbits);
  return ((ZFP_HEADER_MAX_BITS + blocks * maxbits + stream_word_bits - 1) & ~(stream_word_bits - 1)) / CHAR_BIT;
}

void
zfp_stream_rewind(zfp_stream* zfp)
{
  stream_rewind(zfp->stream);
}

void
zfp_stream_set_bit_stream(zfp_stream* zfp, bitstream* stream)
{
  zfp->stream = stream;
}

double
zfp_stream_set_accuracy(zfp_stream* zfp, double tolerance)
{
  int emin = ZFP_MIN_EXP;
  if (tolerance > 0) {
    /* tolerance = x * 2^emin, with 0.5 <= x < 1 */
    frexp(tolerance, &emin);
    emin--;
    /* assert: 2^emin <= tolerance < 2^(emin+1) */
  }
  zfp->minbits = ZFP_MIN_BITS;
  zfp->maxbits = ZFP_MAX_BITS;
  zfp->maxprec = ZFP_MAX_PREC;
  zfp->minexp = emin;
  return tolerance > 0 ? ldexp(1.0, emin) : 0;
}

zfp_field*
zfp_field_alloc()
{
  zfp_field* field = (zfp_field*)malloc(sizeof(zfp_field));
  if (field) {
    field->type = zfp_type_none;
    field->nx = field->ny = field->nz = field->nw = 0;
    field->sx = field->sy = field->sz = field->sw = 0;
    field->data = 0;
  }
  return field;
}

zfp_field*
zfp_field_3d(void* data, zfp_type type, size_t nx, size_t ny, size_t nz)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->ny = ny;
    field->nz = nz;
    field->data = data;
  }
  return field;
}

zfp_field*
zfp_field_1d(void* data, zfp_type type, size_t nx)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->data = data;
  }
  return field;
}

void
zfp_field_free(zfp_field* field)
{
  free(field);
}
