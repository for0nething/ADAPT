
# Take the pytree path operations from src/tree_util.py from JAX 0.4.13
# Reason: The current JAX version 0.3.25 adopted in the project lacks the necessary manipulation functions of pytree path due to incompatibility
# Note: New processing of FrozenDict in _register_keypaths and _registry to support current requirements

# The overall logic should be same as PyTreeDef::FlattenIntoImpl

from __future__ import annotations

import collections
from dataclasses import dataclass
import difflib
import functools
from functools import partial
import operator as op
import textwrap
from typing import (Any, Callable, Hashable, Iterable, NamedTuple,
                    Optional, TypeVar, Union, overload)
import warnings
import flax
from jax._src import traceback_util
from jax._src.lib import pytree
from jax._src.util import safe_zip
from jax._src.util import unzip2

T = TypeVar("T")
U = TypeVar("U", bound=type[Any])

Leaf = Any
PyTreeDef = pytree.PyTreeDef


@dataclass(frozen=True)
class SequenceKey():
  idx: int
  def __str__(self):
    return f'[{repr(self.idx)}]'

@dataclass(frozen=True)
class DictKey():
  key: Hashable
  def __str__(self):
    return f'[{repr(self.key)}]'

@dataclass(frozen=True)
class GetAttrKey():
  name: str
  def __str__(self):
    return f'.{self.name}'

@dataclass(frozen=True)
class FlattenedIndexKey():
  key: int
  def __str__(self):
    return f'[<flat index {self.key}>]'
BuiltInKeyEntry = Union[SequenceKey, DictKey, GetAttrKey, FlattenedIndexKey]

KeyEntry = TypeVar("KeyEntry", bound=Hashable)
KeyPath = tuple[KeyEntry, ...]



def keystr(keys: KeyPath):
  """Helper to pretty-print a tuple of keys.

  Args:
    keys: A tuple of ``KeyEntry`` or any class that can be converted to string.

  Returns:
    A string that joins all string representations of the keys.
  """
  return ''.join([str(k) for k in keys])


class _RegistryWithKeypathsEntry(NamedTuple):
  flatten_with_keys: Callable[..., Any]
  unflatten_func: Callable[..., Any]


def register_keypaths(
    ty: type[T], handler: Callable[[T], tuple[KeyEntry, ...]]
) -> None:
  """[Deprecated] Register the method to get keypaths for type.

  Please use ``register_pytree_with_keys`` instead.

  Only works if the type was already registered with ``register_pytree_node``.
  """
  warnings.warn(
      (
          "jax.tree_util.register_keypaths is deprecated, and will be removed"
          " in a future release. Please use `register_pytree_with_keys()`"
          " instead."
      ),
      category=FutureWarning,
      stacklevel=2,
  )
  _register_keypaths(ty, handler)

_RegistryEntry = collections.namedtuple("_RegistryEntry", ["to_iter", "from_iter"])
_registry = {
    tuple: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: tuple(xs)),
    list: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: list(xs)),
    dict: _RegistryEntry(lambda xs: unzip2(sorted(xs.items()))[::-1],
                         lambda keys, xs: dict(zip(keys, xs))),

    # Changes compared to JAX 0.4.13
    # flax.core.FrozenDict:_RegistryEntry(lambda xs: unzip2(sorted(xs.items()))[::-1],
    flax.core.FrozenDict:_RegistryEntry(lambda xs: unzip2((xs.items()))[::-1],
    # flax.core.FrozenDict:_RegistryEntry(lambda xs: xs.items(),
                         lambda keys, xs: dict(zip(keys, xs))),
    # flax.core.frozen_dict.FrozenDict:_RegistryEntry(lambda xs: unzip2(sorted(xs.items()))[::-1],
    flax.core.frozen_dict.FrozenDict:_RegistryEntry(lambda xs: unzip2(xs.items())[::-1],
    # flax.core.frozen_dict.FrozenDict:_RegistryEntry(lambda xs: xs.items(),
                         lambda keys, xs: dict(zip(keys, xs))),

    type(None): _RegistryEntry(lambda z: ((), None), lambda _, xs: None),

}

def _register_keypaths(
    ty: type[T], handler: Callable[[T], tuple[KeyEntry, ...]]
) -> None:
  def flatten_with_keys(xs):
    children, treedef = _registry[ty].to_iter(xs)
    return list(zip(handler(xs), children)), treedef
  if ty in _registry:
    _registry_with_keypaths[ty] = _RegistryWithKeypathsEntry(
        flatten_with_keys, _registry[ty].from_iter
    )


_registry_with_keypaths = {}

# Changes compared to JAX 0.4.13
_register_keypaths(flax.core.frozen_dict.FrozenDict, lambda xs: tuple(DictKey(k) for k in sorted(xs)))
_register_keypaths(flax.core.FrozenDict, lambda xs: tuple(DictKey(k) for k in sorted(xs)))

_register_keypaths(
    tuple, lambda xs: tuple(SequenceKey(i) for i in range(len(xs)))
)
_register_keypaths(
    list, lambda xs: tuple(SequenceKey(i) for i in range(len(xs)))
)
_register_keypaths(dict, lambda xs: tuple(DictKey(k) for k in sorted(xs)))

_register_keypaths(
    collections.defaultdict, lambda x: tuple(DictKey(k) for k in x.keys())
)

_register_keypaths(
    collections.OrderedDict, lambda x: tuple(DictKey(k) for k in x.keys())
)


def generate_key_paths(
    tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None
) -> list[tuple[KeyPath, Any]]:
  return list(_generate_key_paths_((), tree, is_leaf))
_generate_key_paths = generate_key_paths  # alias for backward compat

def _generate_key_paths_(
    key_path: KeyPath,
    tree: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Iterable[tuple[KeyPath, Any]]:
  if is_leaf and is_leaf(tree):
    yield key_path, tree
    return
  key_handler = _registry_with_keypaths.get(type(tree))
  handler = _registry.get(type(tree))
  if key_handler:
    key_children, _ = key_handler.flatten_with_keys(tree)
    for k, c in key_children:
      yield from _generate_key_paths_((*key_path, k), c, is_leaf)
  elif handler:
    children, _ = handler.to_iter(tree)
    for i, c in enumerate(children):
      k = FlattenedIndexKey(i)
      yield from _generate_key_paths_((*key_path, k), c, is_leaf)
  elif isinstance(tree, tuple) and hasattr(tree, '_fields'):
    # handle namedtuple as a special case, based on heuristic
    key_children = [(GetAttrKey(s), getattr(tree, s)) for s in tree._fields]
    for k, c in key_children:
      yield from _generate_key_paths_(tuple((*key_path, k)), c, is_leaf)
  else:
    yield key_path, tree  # strict leaf type



def tree_flatten(tree: Any,
                 is_leaf: Optional[Callable[[Any], bool]] = None
                 ) -> tuple[list[Leaf], PyTreeDef]:
  """Flattens a pytree.

  The flattening order (i.e. the order of elements in the output list)
  is deterministic, corresponding to a left-to-right depth-first tree
  traversal.

  Args:
    tree: a pytree to flatten.
    is_leaf: an optionally specified function that will be called at each
      flattening step. It should return a boolean, with true stopping the
      traversal and the whole subtree being treated as a leaf, and false
      indicating the flattening should traverse the current object.
  Returns:
    A pair where the first element is a list of leaf values and the second
    element is a treedef representing the structure of the flattened tree.
  """
  return pytree.flatten(tree, is_leaf)



def tree_flatten_with_path(
    tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None
) -> tuple[list[tuple[KeyPath, Any]], PyTreeDef]:
  """Flattens a pytree like ``tree_flatten``, but also returns each leaf's key path.

  Args:
    tree: a pytree to flatten. If it contains a custom type, it must be
      registered with ``register_pytree_with_keys``.
  Returns:
    A pair which the first element is a list of key-leaf pairs, each of
    which contains a leaf and its key path. The second element is a treedef
    representing the structure of the flattened tree.
  """
  _, tree_def = tree_flatten(tree, is_leaf)
  return _generate_key_paths(tree, is_leaf), tree_def


def tree_map_with_path(f: Callable[..., Any],
                       tree: Any, *rest: Any,
                       is_leaf: Optional[Callable[[Any], bool]] = None) -> Any:
  """Maps a multi-input function over pytree key path and args to produce a new pytree.

  This is a more powerful alternative of ``tree_map`` that can take the key path
  of each leaf as input argument as well.

  Args:
    f: function that takes ``2 + len(rest)`` arguments, aka. the key path and
      each corresponding leaves of the pytrees.
    tree: a pytree to be mapped over, with each leaf's key path as the first
      positional argument and the leaf itself as the second argument to ``f``.
    *rest: a tuple of pytrees, each of which has the same structure as ``tree``
      or has ``tree`` as a prefix.

  Returns:
    A new pytree with the same structure as ``tree`` but with the value at each
    leaf given by ``f(kp, x, *xs)`` where ``kp`` is the key path of the leaf at
    the corresponding leaf in ``tree``, ``x`` is the leaf value and ``xs`` is
    the tuple of values at corresponding nodes in ``rest``.
  """

  keypath_leaves, treedef = tree_flatten_with_path(tree, is_leaf)
  keypath_leaves = list(zip(*keypath_leaves))
  all_keypath_leaves = keypath_leaves + [treedef.flatten_up_to(r) for r in rest]
  return treedef.unflatten(f(*xs) for xs in zip(*all_keypath_leaves))

