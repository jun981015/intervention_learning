"""Observation encoder modules.

Encoders transform raw observations into features for networks. Keep
environment wrappers in `envs` and loss/update logic in `algo`.
"""

from il.networks.encoders.d4pg_encoder import D4PGEncoder

__all__ = ["D4PGEncoder"]
