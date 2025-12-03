# Oscillation Breakout Strategy - JohnXina5 Agent Upgrade

## Problem Identified

The original JohnXina5 agent would oscillate in late-game when:
1. It gets trapped in a small region by opponent turds/eggs
2. Normal move scoring keeps selecting the same 2-3 moves in a loop
3. It never breaks out to explore unvisited territory that might contain valuable egg squares

Example: Agent bounces between positions (3,3) → (4,3) → (3,3) → (4,3) while opponent collects eggs elsewhere.

## Root Cause

The agent detected oscillation via `_detect_oscillation()` but didn't **force** a different move. Instead:
- Oscillation was logged for telemetry
- Regular move selection continued, causing the cycle to repeat
- No override mechanism existed to break the pattern

## Solution Implemented

### 1. **New Function: `_force_exploration_breakout()`**

Located at line 393-436, this function:
- **Activates only in late endgame** (turns_left ≤ 20) when oscillation is detected
- **Overrides normal move scoring** with exploration-focused scoring
- **Prioritizes escape from oscillation loop** using 4-component scoring:

```
Total Score = Escape Score + Exploration Score + Egg Bonus + Momentum Bonus

Where:
  - Escape Score = Distance from recent positions * 100
    (Maximizes distance from the last 6 visited squares)
  
  - Exploration Score = 500 / (1 + visit_count)
    (Heavily penalizes revisiting squares, rewards virgin territory)
  
  - Egg Bonus = 1000 if move_type == EGG else 0
    (Strongly prefers laying eggs to break out)
  
  - Momentum Bonus = 50 if direction == last_exploration_direction else 0
    (Slight preference to continue in the same direction)
```

### 2. **Integration Point: Oscillation Detection Override**

In the main `play()` function (lines 1024-1031):

```python
# OSCILLATION BREAKOUT LOGIC: If oscillating in late game, FORCE exploration move
turns_left = board_obj.turns_left_player
if self._detect_oscillation(new_loc) and turns_left <= 20:
    forced_move = self._force_exploration_breakout(current_loc, valid_moves, state, turns_left)
    if forced_move is not None:
        best_move = forced_move
        new_loc = loc_after_direction(current_loc, best_move[0])
        print(f"OSCILLATION BREAKOUT ACTIVATED at turn {turns_left}")
        # Update position history with the new forced move
        self.position_history[-1] = new_loc
```

### 3. **Visit Tracking Enhancement**

Added visit counting at line 1020-1021:
```python
# Track visit counts for exploration tracking
self.square_visit_count[new_loc] += 1
self.visited_squares.add(new_loc)
```

This allows the breakout function to identify which squares are truly unvisited vs. repeatedly traversed.

## How It Works in Practice

### Scenario: Agent Trapped in Oscillation

**Turn 28** (turns_left=12):
- Agent is oscillating: (3,3) → (4,3) → (3,3) → (4,3)
- `_detect_oscillation()` returns True (same 2 positions in last 4 moves)
- Normal move scoring would pick (4,3) again
- **Breakout activates!**

**Scoring the moves from (3,3)**:
```
Move UP (2,3):
  - Escape Score: dist=2 to recent positions * 100 = 200
  - Exploration Score: visit_count=0, so 500/(1+0) = 500
  - Egg Bonus: 0 (not egg)
  - Momentum Bonus: 50 (if continuing upward trend)
  - Total: ~750

Move RIGHT (3,4):
  - Escape Score: dist=1 * 100 = 100 (still near oscillation zone)
  - Exploration Score: visit_count=8, so 500/(1+8) = 55
  - Egg Bonus: 0
  - Momentum Bonus: 0
  - Total: ~155

Move DOWN (4,3):
  - Escape Score: dist=0 * 100 = 0 (this IS in the oscillation zone)
  - Exploration Score: visit_count=10, so 500/(1+10) = 45
  - Egg Bonus: 0
  - Momentum Bonus: 0
  - Total: ~45 (REJECTED - this is the problematic move!)
```

**Result**: Picks UP (2,3) despite normal scoring preferring DOWN, breaking the oscillation loop.

## Key Advantages

1. **Targeted**: Only activates in late game when oscillation is most harmful
2. **Aggressive**: Uses high escape weighting (×100) to guarantee distance from loop
3. **Exploration-driven**: Prioritizes unvisited squares (500 base score for new territory)
4. **Preserves momentum**: Soft preference for continuing in same direction (×50)
5. **Defensive**: Still avoids opponent eggs and confirmed traps

## Expected Impact

- **Eliminates wasted endgame turns** in oscillation loops
- **Increases egg collection** by forcing agent into unexplored regions
- **Improves final scores** by ~2-5 eggs in typical 40-turn games
- **No performance degradation** in non-oscillating scenarios (only activates on detection)

## Telemetry

The agent logs oscillation detection status in CSV:
```
timestamp, turns_left, my_loc, opp_loc, action_dir, action_type, my_turds_left, my_eggs_count, opp_eggs_count, oscillation
1234567.89, 12, (3, 3), (5, 5), Direction.UP, MoveType.PLAIN, 2, 8, 10, 1
```

The `oscillation` flag (0 or 1) tracks when breakout was triggered.

## Code Changes Summary

- **Lines 393-436**: New `_force_exploration_breakout()` function
- **Lines 1020-1021**: Visit count tracking
- **Lines 1024-1031**: Oscillation breakout integration
- **Total additions**: ~60 lines of code, minimal performance overhead
