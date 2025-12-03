# Oscillation Breakout Logic - Test Scenarios

## Test Case 1: Standard Oscillation Breakout

**Situation:**
- Current location: (3, 3)
- Position history (last 6): [(3,3), (4,3), (3,3), (4,3), (3,3), (4,3)]
- Valid moves available: UP→(2,3), RIGHT→(3,4), DOWN→(4,3), LEFT→(3,2)
- Turns left: 12

**Move Scoring:**

1. **UP to (2,3)** (not in recent history)
   - recent_positions = {(3,3), (4,3)}
   - min_dist_to_recent = min(|2-3|+|3-3|, |2-4|+|3-3|) = min(1, 2) = 1
   - escape_score = 1 × 100 = 100
   - visit_count = 0 (first time here)
   - exploration_score = 500 / (1+0) = 500
   - egg_bonus = 0 (assuming PLAIN move)
   - momentum_bonus = 50 (if UP was last direction)
   - **TOTAL = 650**

2. **RIGHT to (3,4)** (rarely visited)
   - min_dist_to_recent = min(|3-3|+|4-3|, |3-4|+|4-3|) = min(1, 2) = 1
   - escape_score = 1 × 100 = 100
   - visit_count = 5 (revisited area)
   - exploration_score = 500 / (1+5) = 83
   - egg_bonus = 0
   - momentum_bonus = 0
   - **TOTAL = 183**

3. **DOWN to (4,3)** (IN the oscillation zone!)
   - min_dist_to_recent = min(|4-3|+|3-3|, |4-4|+|3-3|) = min(1, 0) = 0
   - escape_score = 0 × 100 = 0
   - visit_count = 10 (highly revisited)
   - exploration_score = 500 / (1+10) = 45
   - egg_bonus = 0
   - momentum_bonus = 0
   - **TOTAL = 45** ← REJECTED

4. **LEFT to (3,2)** (partially explored)
   - min_dist_to_recent = min(|3-3|+|2-3|, |3-4|+|2-3|) = min(1, 2) = 1
   - escape_score = 1 × 100 = 100
   - visit_count = 2
   - exploration_score = 500 / (1+2) = 167
   - egg_bonus = 0
   - momentum_bonus = 0
   - **TOTAL = 267**

**Result**: UP (2,3) with score 650 ✅
- Breaks oscillation by moving to unexplored square
- Maximizes distance from oscillation zone

---

## Test Case 2: Egg Breakout Priority

**Situation:**
- Current location: (5, 5)
- Recent oscillation between (5,5) and (6,5)
- Valid moves: UP→(4,5) [PLAIN], RIGHT→(5,6) [EGG], DOWN→(6,5) [PLAIN], LEFT→(5,4) [PLAIN]
- Turns left: 15

**Move Scoring:**

1. **UP to (4,5)** [PLAIN]
   - escape_score = 150 (distance 1.5 from recent)
   - exploration_score = 400 (fairly unvisited)
   - egg_bonus = 0
   - **TOTAL ≈ 550**

2. **RIGHT to (5,6)** [EGG] ✨
   - escape_score = 150
   - exploration_score = 350 (somewhat visited)
   - egg_bonus = 1000 ← EGG MOVE!
   - **TOTAL ≈ 1500** ← HIGHEST!

3. **DOWN to (6,5)** [PLAIN] (oscillation zone)
   - escape_score = 0
   - exploration_score = 50
   - egg_bonus = 0
   - **TOTAL ≈ 50**

4. **LEFT to (5,4)** [PLAIN]
   - escape_score = 150
   - exploration_score = 300
   - egg_bonus = 0
   - **TOTAL ≈ 450**

**Result**: RIGHT (5,6) with EGG move, score 1500 ✅
- Even though not furthest away, egg placement bonus dominates
- Effectively "breaks out" by laying egg (commits territory)

---

## Test Case 3: Empty Breakout (All moves lead to bad locations)

**Situation:**
- Current location: (0, 0) - corner with no good moves
- All adjacent squares: opponent eggs or confirmed traps
- Valid moves: []

**Expected Behavior:**
```python
if turns_left > 20 or not valid_moves:
    return None  # Only engage in late endgame
```
- Function returns None immediately
- Calling code checks `if forced_move is not None:`
- Falls back to original best_move from normal scoring
- **Safety maintained** ✅

---

## Test Case 4: No Oscillation Detected

**Situation:**
- Position history shows varied movement (not oscillating)
- `_detect_oscillation()` returns False

**Expected Behavior:**
```python
if self._detect_oscillation(new_loc) and turns_left <= 20:
    forced_move = self._force_exploration_breakout(...)
    # THIS BLOCK NEVER EXECUTES
```
- Breakout function never called
- Normal move selection continues
- **Zero overhead for non-oscillating games** ✅

---

## Test Case 5: Early Game (turns_left > 20)

**Situation:**
- Oscillation detected at turn 25 (turns_left=15) ✓
- Oscillation detected at turn 10 (turns_left=30) ✗

**Expected Behavior:**
```python
if turns_left > 20 or not valid_moves:
    return None  # Only engage in late endgame
```

**Late Game Oscillation (turn 25, turns_left=15):**
- Condition: `15 <= 20` → True
- Breakout engages ✅

**Early Game Oscillation (turn 10, turns_left=30):**
- Condition: `30 <= 20` → False
- Breakout disengaged
- Normal scoring handles it
- **Allows flexibility in early game** ✅

---

## Scoring Sensitivity Analysis

The formula prioritizes:
1. **Escape from loop** (×100 multiplier) - highest priority
2. **Exploration of new territory** (×500 base) - second priority
3. **Egg placement** (×1000 bonus) - strategic advantage
4. **Direction momentum** (×50) - minor convenience

Example with two moves both with escape_score=100:
- Move A: no visit, no egg → 500
- Move B: 5 visits, is egg → 500/(1+5) + 1000 = 1083
- **Egg bonus completely dominates** → ensures breakout is impactful

---

## Integration Verification

Critical checkpoints in code flow:

1. ✅ Position history updated BEFORE oscillation check
2. ✅ Visit counts tracked for all moves
3. ✅ Oscillation detection called with current new_loc
4. ✅ Breakout only engages in late game (turns_left ≤ 20)
5. ✅ Forced move overrides previous best_move
6. ✅ Position history updated with forced move location
7. ✅ Exploration momentum maintained for next turn

All safety checks in place!
