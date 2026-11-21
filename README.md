# Cyber_Ai — Mech Royale bot

A `mechfight/1` bot for the Builders Buildathon Mech Royale track.

- `bot.py` — the bot: zone-aware movement, A* navigation, target leading, weapon handling (gun, lightning, laser), dodging, looting and upgrades.
- `world.py` — static map model: terrain heights, building solids for line of sight, 2 m walk grid with a resumable A*.
- `data/` — grids precomputed offline from the kit's `arena/v2` map (see `tools/precompute.py` in the dev folder).
- `mechfight.py` — the kit's Python wrapper (unchanged).

Python standard library only. Pre-existing code used: the kit's `mechfight.py` wrapper. Released under the MIT License.
