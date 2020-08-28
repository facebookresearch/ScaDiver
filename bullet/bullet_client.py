# Copyright (c) Facebook, Inc. and its affiliates. 

'''
This class was slightly edited from the code existing in one of examples in PyBullet.
'''

import functools
import inspect
import pybullet

class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=pybullet.DIRECT, options=""):
        """Create a simulation and connect to it."""
        self._client = pybullet.connect(pybullet.SHARED_MEMORY)
        if (self._client < 0):
            self._client = pybullet.connect(connection_mode, options=options)
        self._shapes = {}

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute
