"""
This is a module for defining private helpers which do not depend on the
rest of cgm.

Everything in here must be self-contained so that it can be
imported anywhere else without creating circular imports.
If a utility requires the import of cgm, it probably belongs
in ``cgm.core``.

This is based on numpy._utils.__init__.py.
"""

def set_module(module):
    """Private decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module('cgm')
        def example():
            pass

        assert example.__module__ == 'cgm'
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator
