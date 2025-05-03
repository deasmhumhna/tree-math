from string import ascii_lowercase
from operator import xor

from typing import Any, Sequence, List

from functools import partial

from tree_math import Vector, VectorMixin
from jax import tree_util

from dataclasses import field, dataclass

from math import prod
from jax._src.lib import pytree


PyTreeDef = pytree.PyTreeDef
leaf_pytree = jax.tree.structure(0)
default_registry = pytree.default_registry()


def is_wrapped(tree, outer_treedef):
    return jax.tree.structure(tree).node_data() != outer_treedef.node_data()


def wrap_treedefs(parent, child_treedefs):
    parent_treedef = jax.tree.structure(parent)
    node_data = parent_treedef.node_data()
    assert len(child_treedefs) == len(parent_treedef.children())
    return parent_treedef.make_from_node_data_and_children(
        default_registry, node_data, child_treedefs
    )


def tree_matrix_to_array(x, outer_treedefs, outer_ndims):
    if len(outer_treedefs) > 1 or is_wrapped(x, outer_treedefs[0]):
        outer_treedef = wrap_treedefs(x, outer_treedefs)
    else:
        outer_treedef = outer_treedefs[0]
    subtrees = outer_treedef.flatten_up_to(x)
    rows = []
    for subtree, outer_ndim in zip(subtrees, outer_ndims):
        leaves, treedef = jax.tree.flatten(subtree)
        if outer_ndim > 0:
            leading_size = prod(jnp.shape(leaves[0])[:outer_ndim])
            flat = [jnp.reshape(leaf, jnp.shape(leaf)[:outer_ndim] + (-1,)) for leaf in leaves]
            rows.extend([jax.tree.unflatten(treedef, (elem[i] for elem in flat)) for i in range(leading_size)])
        else:
            rows.append(subtree)
    return jnp.stack([jax.flatten_util.ravel_pytree(row)[0]for row in rows], axis=0)


def get_leaves_from_wrapped(tree, outer_treedef):
    node_data = jax.tree.structure(tree).node_data()
    outer_treedef = outer_treedef.make_from_node_data_and_children(
        default_registry, node_data, [outer_treedef]
    )
    return outer_treedef.flatten_up_to(tree)


def eindot(x, y, x_row_ndim=None, y_col_ndim=None, precision='highest'):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x_row_ndim is not None:
        x_col_ndim = x.ndim - x_row_ndim
        ein1 = ascii_lowercase[:x_row_ndim]
        ein2 = ascii_lowercase[x_row_ndim:x_row_ndim + y.ndim - x_col_ndim]
        ein3 = ascii_lowercase[:x_col_ndim].upper()
    elif y_col_ndim is not None:
        y_row_ndim = y.ndim - y_col_ndim
        ein1 = ascii_lowercase[y_col_ndim:y_col_ndim + x.ndim - y_row_ndim]
        ein2 = ascii_lowercase[:y_col_ndim]
        ein3 = ascii_lowercase[:y_row_ndim].upper()
    subscripts = "{0}{2}, {2}{1} -> {0}{1}".format(ein1, ein2, ein3)
    return jnp.einsum(subscripts, x, y, precision=precision)


def product_sum(x: Any, y: Any, x_treedefs: Sequence[PyTreeDef], y_treedefs: Sequence[PyTreeDef], x_outer_ndim: int, y_outer_ndims: Sequence[int], precision='highest') -> Any:
    # (a inner structure) == (b outer structure)
    # result = (a outer structure) x (b inner structure)
    if len(x_treedefs) > 1 or is_wrapped(x, x_treedefs[0]):
        x_treedef = wrap_treedefs(x, x_treedefs)
    else:
        x_treedef = x_treedefs[0]
    if len(y_treedefs) > 1 or is_wrapped(y, y_treedefs[0]):
        y_treedef = wrap_treedefs(y, y_treedefs)
    else:
        y_treedef = y_treedefs[0]

    try:
        x_in_leaves = x_treedef.flatten_up_to(x)
        y_in_trees  = y_treedef.flatten_up_to(y)
    except ValueError:
        raise ValueError("incompatible structures")

    out_trees = []
    for x_leaf, y_tree in zip(x_in_leaves, y_in_trees):
        out_trees.append(jax.tree.map(partial(eindot, x_leaf, x_row_ndim=x_outer_ndim, precision=precision), y_tree))

    return jax.tree.map(lambda *x: sum(x), *out_trees)


def dot_product(x: Any, y: Any, x_treedefs: Sequence[PyTreeDef], y_treedefs: Sequence[PyTreeDef], x_outer_ndim: int, y_outer_ndims: Sequence[int], precision='highest') -> Any:
    # share same inner structure
    # result = (a outer structure) x (b outer structure)
    if len(x_treedefs) > 1 or is_wrapped(x, x_treedefs[0]):
        x_treedef = wrap_treedefs(x, x_treedefs)
    else:
        x_treedef = x_treedefs[0]
    if len(y_treedefs) > 1:
        y_treedef = wrap_treedefs(y, y_treedefs)
        out_treedef = y_treedef
    elif is_wrapped(y, y_treedefs[0]):
        y_treedef = wrap_treedefs(y, y_treedefs)
        out_treedef = y_treedefs[0]
    else:
        y_treedef = y_treedefs[0]
        out_treedef = y_treedef

    try:
        x_leaves = x_treedef.flatten_up_to(x)
        y_trees = y_treedef.flatten_up_to(y)
    except ValueError:
        raise ValueError("incompatible structures")

    out_leaves = []
    for y_tree, y_outer_ndim in zip(y_trees, y_outer_ndims):
        y_leaves = jax.tree.leaves(y_tree)
        out_leaf = sum(eindot(x_leaf, y_leaf, y_col_ndim=y_outer_ndim, precision=precision) for x_leaf, y_leaf in zip(x_leaves, y_leaves))
        out_leaves.append(out_leaf)
    return jax.tree.unflatten(out_treedef, out_leaves)


def outer_product_sum(
    a: Any,
    b: Any,
    a_outer_treedef: PyTreeDef,
    b_outer_treedef: PyTreeDef,
    a_outer_ndims: Sequence[int],
    precision='highest'
) -> Any:

    # share same outer structure
    # result = (a inner structure) x (b inner structure
    a_in_trees = a_outer_treedef.flatten_up_to(a)
    b_in_trees = b_outer_treedef.flatten_up_to(b)

    out_trees = []
    for a_tree, b_tree, a_outer_ndim in zip(a_in_trees, b_in_trees, a_outer_ndims):
        out_leaves = []
        a_in_leaves, a_treedef = jax.tree.flatten(a_tree)
        # wrap a_treedef
        for a_leaf in a_in_leaves:
            a_inner_ndim = jnp.asarray(a_leaf).ndim - a_outer_ndim
            out_leaves.append(jax.tree.map(partial(eindot, a_leaf, x_row_ndim=a_inner_ndim, precision=precision), b_tree))
        out_trees.append(jax.tree.unflatten(a_treedef, out_leaves))
    return jax.tree.map(lambda *x: sum(x), *out_trees)


def matmul(
    a: Any,
    b: Any,
    a_outer_treedefs: List[PyTreeDef],
    b_outer_treedefs: List[PyTreeDef],
    a_outer_ndims: Sequence[int],
    b_outer_ndims: Sequence[int],
    trans_a: bool,
    trans_b: bool
) -> Any:
    """
    Matrix-vector or matrix-matrix multiplication between tree math objects.

    Note that unlike jax.numpy.matmul, tree_math.matmul defaults to full (highest)
    precision. This is more useful for numerical algorithms and will be the
    default for jax.numpy in the future:
    https://github.com/google/jax/pull/7859

    Args:
      a: left argument.
      b: right argument.
      a_outer_treedefs: List of pytrees specifying the outer tree stucture of the left matrix-like object.
      b_outer_treedefs: List of pytrees specifying the outer tree stucture of the right matrix-like object.
      precision: precision.

    """
    if is_wrapped(a, a_outer_treedefs[0]):
        a_outer_treedef = wrap_treedefs(a, a_outer_treedefs)
    else:
        a_outer_treedef = a_outer_treedefs[0]
    if is_wrapped(b, b_outer_treedefs[0]):
        b_outer_treedef = wrap_treedefs(b, b_outer_treedefs)
    else:
        b_outer_treedef = b_outer_treedefs[0]

    in_leaves = a_outer_treedef.flatten_up_to(a)
    a_inner_treedef = jax.tree.structure(in_leaves[0])

    if trans_a and not trans_b:
        out = outer_product_sum(a, b, a_outer_treedef, b_outer_treedef, a_outer_ndims)
        leaves = a_inner_treedef.flatten_up_to(out)
        if len(a_outer_treedefs) == len(a_inner_treedef.children()):
            new_treedef = a_outer_treedef.make_from_node_data_and_children(
                default_registry, a_outer_treedef.node_data(), a_inner_treedef.children()
            )
        elif len(b_outer_treedefs) == len(a_inner_treedef.children()):
            new_treedef = b_outer_treedef.make_from_node_data_and_children(
                default_registry, b_outer_treedef.node_data(), a_inner_treedef.children()
            )
        elif len(a_outer_treedefs) == 1:
            new_treedef = a_outer_treedef.make_from_node_data_and_children(
                default_registry, a_outer_treedef.node_data(), [a_inner_treedef]
            )
        elif len(b_outer_treedefs) == 1:
            new_treedef = b_outer_treedef.make_from_node_data_and_children(
                default_registry, b_outer_treedef.node_data(), [a_inner_treedef]
            )
        else:
            return out, False
        return jax.tree.unflatten(new_treedef, leaves), False
    if trans_b and not trans_a:
        inner = dot_product
    else:
        if trans_a:
            a, b = b, a
            a_outer_treedef, b_outer_treedef = b_outer_treedef, a_outer_treedef
            a_outer_ndims, b_outer_ndims = b_outer_ndims, a_outer_ndims
        inner = product_sum

    out_leaves = []
    for in_leaf, a_outer_ndim in zip(in_leaves, a_outer_ndims):
        out_leaf = inner(in_leaf, b, [a_inner_treedef], b_outer_treedefs, a_outer_ndim, b_outer_ndims)
        out_leaves.append(out_leaf)
    return jax.tree.unflatten(a_outer_treedef, out_leaves), (trans_a and trans_b)


class MatrixMixin(VectorMixin):
    """
    A mixin class that adds a 2D matrix-like behaviour to any properly defined custom pytree class.

    Tree math matrix can be any pytree of pytrees with consistant inner pytree tree stucture. For example,

        {a: [1, 2], b: [3, 4]}

    could be a tree math matrix with an outer tree structure of {a: *, b: *} and an inner tree structure of [*, *]. However,

        {a: [1, 2], b: {3, 4}}

    cannot be a tree math matrix because the leaves of the outer tree have different inner structure.

    outer_treedefs: Sequence[PyTreeDef]. A Sequence of PyTreeDef objects that describe the outer tree structure(s). Usually a single PyTreeDef. However, in the case of mixin classes,
    you must provide the outer treedef for each data field. These are automatically combined with a cls node to create the full outer tree of the object.

    outer_ndims: Sequence[int]. For each leaf in the outer_treedef, outer_ndims holds the number of leading dimensions for which slices should be interpreted as separate rows within the corresponding inner tree.
    For a leaf with shape [2, 3, 4], an outer_ndim for 1 means this leaf hold row with 12 values each, while an outer_ndim of 2 means this leaves holds 6 rows of 4.
    You must list outer_ndims in the outer tree flattening order. All leaves within an inner tree must have the same shape in all dimensions specified by the outer_ndim.

    trans: Flag to denote if this object should be interpreted as the transpose of the matrix described by outer_treedefs and outer_ndims. Only used to choose the appropriate matrix multiplication algorithm
    """

    outer_treedefs: Sequence[PyTreeDef]
    outer_ndims: Sequence[int]
    trans: bool = False

    @property
    def size(self):
      return prod(self.shape)

    def __len__(self):
      return self.shape[0]

    @property
    def shape(self):
        outer_treedef = wrap_treedefs(self, self.outer_treedefs)
        leaves = outer_treedef.flatten_up_to(self)

        outer_size = 0
        for leaf, outer_ndim in zip(leaves, self.outer_ndims):
            values = jax.tree.leaves(leaf)
            outer_size += prod(values[0].shape[:outer_ndim])

        inner_size = sum(prod(value.shape[outer_ndim:]) for value in values)
        return (outer_size, inner_size) if not self.trans else (inner_size, outer_size)

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        values = tree_util.tree_leaves(self)
        return jnp.result_type(*values)

    def to_array(self):
        return tree_matrix_to_array(self, self.outer_treedefs, self.outer_ndims)

    def __matmul__(self, other):
        if isinstance(other, MatrixMixin):
            res_is_matrix = True
            other_outer_treedefs = other.outer_treedefs
            other_outer_ndims = other.outer_ndims
            other_trans = other.trans
        elif isinstance(other, VectorMixin):
            res_is_matrix = False
            treedef = jax.tree.structure(other)
            other_outer_treedefs = treedef.children() if not self.trans else [leaf_pytree]
            other_outer_ndims = [0] * sum(tree.num_leaves for tree in other_outer_treedefs)
            other_trans = self.trans
        else:
            raise NotImplementedError

        res, trans = matmul(self, other, self.outer_treedefs, other_outer_treedefs, self.outer_ndims, other_outer_ndims, self.trans, other_trans)

        if not (res_is_matrix or self.trans):
            if is_wrapped(self, self.outer_treedefs[0]):
                treedef = wrap_treedefs(self, self.outer_treedefs)
            else:
                treedef = self.outer_treedefs[0]
            leaves = treedef.flatten_up_to(res)
            other_treedef = jax.tree.structure(other)
            if len(other_treedef.children()) > 1 or len(self.outer_treedefs) == 1:
                new_treedef = other_treedef.make_from_node_data_and_children(
                    default_registry, other_treedef.node_data(), self.outer_treedefs
                )
            else:
                # wrap in list first
                sub_treedef = other_treedef.make_from_node_data_and_children(
                    default_registry, (list, None), self.outer_treedefs
                )
                new_treedef = other_treedef.make_from_node_data_and_children(
                    default_registry, other_treedef.node_data(), [sub_treedef]
                )
            return jax.tree.unflatten(new_treedef, leaves)

        return res if not xor(trans, res.trans) else res.T

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        outer_treedef = wrap_treedefs(self, self.outer_treedefs)
        in_leaves = outer_treedef.flatten_up_to(self)

        # get col_ndims and transpose leaves
        out_leaves = []
        for i, (in_leaf, row_ndim) in enumerate(zip(in_leaves, self.outer_ndims)):
            in_values, treedef = jax.tree.flatten(in_leaf)
            if i == 0:
                inner_treedef = treedef
                col_ndims = [value.ndim - row_ndim for value in in_values]
            out_values = []
            for value in in_values:
                permutation = range(row_ndim-value.ndim, row_ndim)
                out_values.append(jnp.transpose(value, permutation))
            out_leaves.append(jax.tree.unflatten(treedef, out_values))

        cls, attrs = outer_treedef.node_data()
        new_attrs = attrs[:2] + (not attrs[2],) + attrs[3:] # flip trans flag
        new_treedef = treedef.make_from_node_data_and_children(
            default_registry, (cls, new_attrs), self.outer_treedefs
        )
        return jax.tree.unflatten(new_treedef, out_leaves)


@partial(jax.tree_util.register_dataclass,
         data_fields=['tree'],
         meta_fields=['outer_treedefs', 'outer_ndims', 'trans'])
class Matrix(MatrixMixin):

    def __init__(self, tree, outer_treedefs, outer_ndims, *, trans=False):
        if not isinstance(outer_treedefs, Sequence):
            outer_treedefs = [outer_treedefs]
        if not isinstance(outer_ndims, Sequence):
            outer_ndims = [outer_ndims] * outer_treedefs[0].num_leaves

        assert not trans
        assert len(outer_treedefs) == 1
        assert len(outer_ndims) == outer_treedefs[0].num_leaves

        self._tree = tree
        self.outer_treedefs = outer_treedefs
        self.outer_ndims = outer_ndims
        self.trans = trans

    @property
    def tree(self):
        return self._tree

    @property
    def outer_treedef(self):
        return self.outer_treedefs[0]

    def transpose(self):
      subtree = self.outer_treedefs[0].flatten_up_to(self._tree)[0]
      inner_treedef = jax.tree.structure(subtree)
      inner_ndims = [jnp.asarray(value).ndim - self.outer_ndims[0] for value in jax.tree.leaves(subtree)]

      # transpose structures
      trans = jax.tree.transpose(self.outer_treedefs[0], inner_treedef, self._tree)
      subtrees = inner_treedef.flatten_up_to(trans)

      # transpose leaves
      trans_subtrees = []
      for i, (subtree, row_ndim) in enumerate(zip(subtrees, inner_ndims)):
          leaves, treedef = jax.tree.flatten(subtree)
          trans_leaves = []
          for leaf in leaves:
              permutation = range(row_ndim - jnp.asarray(leaf).ndim, row_ndim)
              trans_leaves.append(jnp.transpose(leaf, permutation))
          trans_subtrees.append(jax.tree.unflatten(treedef, trans_leaves))

      # update metadata
      treedef = jax.tree.structure(self)
      cls, data = treedef.node_data()
      data = (inner_treedef, inner_ndims) + data[2:]
      trans_treedef = treedef.make_from_node_data_and_children(
          default_registry, (cls, data), [inner_treedef]
      )
      trans = jax.tree.unflatten(trans_treedef, trans_subtrees)
      return trans

    def __repr__(self):
        return f"tree_math.Matrix({self._tree!r})"


@partial(jax.tree_util.register_dataclass,
         data_fields=['tree'],
         meta_fields=['outer_treedefs', 'outer_ndims', 'trans'])
class VectorStack(Matrix):

    def __init__(self, tree):
        self._tree = tree
        outer_treedefs = [leaf_pytree]
        outer_ndims = [1]
        trans = False

    def __getitem__(self, index: int):
        return Vector(jax.tree.map(lambda x: x[index], self._tree))

    def __setitem__(self, index: int, other: VectorMixin):
        self_treedef, self_leaves = jax.tree.flatten(self._tree)
        other_leaves = jax.tree.leaves(other)
        out_leaves = list(map(lambda x, y: x.at[index].set(y), self_leaves, other_leaves))
        return jax.tree.unflatten(self_treedef, out_leaves)

    def __repr__(self):
        return f"tree_math.VectorStack({self._tree!r})"
