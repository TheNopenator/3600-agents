from collections.abc import Callable
from collections import defaultdict, deque
from time import time
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import math
import random
import copy
import csv
import os

from game.board import Board, manhattan_distance
from game.enums import Direction, MoveType, loc_after_direction
from game.game_map import prob_hear, prob_feel

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move 
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
    
    def ucb_score(self, exploration=1.4):
        if self.visits == 0:
            return float('inf')
        return (self.total_reward / self.visits) + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

# Lightweight MCTSNode stub: keep minimal shape but remove reliance on heavy rollout logic
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.total_reward = 0.0

    def ucb_score(self, exploration=1.4):
        if self.visits == 0:
            return float('inf')
        # safe fallback for parent visits
        parent_visits = max(1, getattr(self.parent, 'visits', 1))
        return (self.total_reward / max(1, self.visits)) + exploration * math.sqrt(
            math.log(parent_visits) / max(1, self.visits)
        )

class TimeoutError(Exception):
    """Custom exception raised when search budget is exceeded."""
    pass

class TrapdoorTracker:
    def __init__(self, map_size: int = 8):
        self.map_size = map_size
        # Separate belief grids for white (even parity) and black (odd parity) trapdoors
        self.white_belief = self._initialize_belief_grid()  # For trapdoor on white squares
        self.black_belief = self._initialize_belief_grid()  # For trapdoor on black squares
        
        # Confirmed locations
        self.confirmed_white_trap = None
        self.confirmed_black_trap = None
        
    def _initialize_belief_grid(self) -> np.ndarray:
        """Initialize prior belief over trapdoor locations based on rule weighting."""
        belief = np.zeros((self.map_size, self.map_size))
        
        # Rule: weight by distance from edge
        # Edge (0 or 7): weight 0
        # 1 step in (1 or 6): weight 0
        # 2 steps in (2 or 5): weight 1
        # 3 steps in (3 or 4): weight 2 (center area)
        for r in range(self.map_size):
            for c in range(self.map_size):
                dist_from_edge = min(r, c, self.map_size - 1 - r, self.map_size - 1 - c)
                if dist_from_edge == 0 or dist_from_edge == 1:
                    belief[r, c] = 0.0
                elif dist_from_edge == 2:
                    belief[r, c] = 1.0
                elif dist_from_edge >= 3:
                    belief[r, c] = 2.0
        
        # Normalize
        total = belief.sum()
        if total > 0:
            belief /= total
        
        return belief
    
    def update(self, current_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        """
        Bayesian update of trapdoor beliefs given sensor readings.
        
        sensor_data[0] = (heard_white, felt_white) for white square trapdoor
        sensor_data[1] = (heard_black, felt_black) for black square trapdoor
        """
        heard_white, felt_white = sensor_data[0]
        heard_black, felt_black = sensor_data[1]
        
        # Update white trapdoor belief
        self._bayesian_update(self.white_belief, current_loc, heard_white, felt_white)
        
        # Update black trapdoor belief
        self._bayesian_update(self.black_belief, current_loc, heard_black, felt_black)
    
    def _bayesian_update(self, belief_grid: np.ndarray, current_loc: Tuple[int, int], 
                        heard: bool, felt: bool):
        """
        Perform Bayesian update: belief ∝ prior * likelihood
        
        Likelihood is based on distance to current location.
        """
        x, y = current_loc
        
        # Compute likelihood for each possible trapdoor location
        likelihood = np.zeros_like(belief_grid)
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                # Distance metrics
                dx = abs(r - x)
                dy = abs(c - y)
                manhattan_dist = dx + dy
                is_diagonal = (dx == 1 and dy == 1)
                is_adjacent = (manhattan_dist == 1)
                
                # Probability of hearing/feeling if trapdoor IS at (r, c)
                if is_adjacent:
                    # Adjacent (share edge): 50% hear, 30% feel
                    p_hear = 0.5
                    p_feel = 0.3
                elif is_diagonal:
                    # Diagonal: 25% hear, 15% feel
                    p_hear = 0.25
                    p_feel = 0.15
                elif manhattan_dist == 2:
                    # Two steps away: 10% hear, 0% feel
                    p_hear = 0.10
                    p_feel = 0.0
                else:
                    # Far away: essentially 0% both
                    p_hear = 0.0
                    p_feel = 0.0
                
                # Likelihood given observation
                # P(observation | trapdoor at (r,c))
                if heard:
                    l_hear = p_hear
                else:
                    l_hear = 1.0 - p_hear
                
                if felt:
                    l_feel = p_feel
                else:
                    l_feel = 1.0 - p_feel
                
                # Combined likelihood (independent given location)
                likelihood[r, c] = l_hear * l_feel
        
        # Bayesian update: posterior ∝ prior * likelihood
        belief_grid *= likelihood
        
        # Normalize
        total = belief_grid.sum()
        if total > 0:
            belief_grid /= total
        else:
            # If all probabilities became 0 (shouldn't happen), reset to prior
            belief_grid[:] = self._initialize_belief_grid()
    
    def get_risk(self, loc: Tuple[int, int]) -> float:
        """
        Get trapdoor risk at a location.
        
        Returns probability that stepping on this square will trigger a trapdoor.
        Since there are two trapdoors (one white, one black), combine both beliefs.
        """
        r, c = loc
        
        # Bounds check - return 0 risk if out of bounds
        if not (0 <= r < 8 and 0 <= c < 8):
            return 0.0
        
        # If confirmed, return certainty
        if self.confirmed_white_trap == loc:
            return 1.0
        if self.confirmed_black_trap == loc:
            return 1.0
        
        # Get probability from appropriate belief grid based on square parity
        if (r + c) % 2 == 0:
            # White square - risk from white trapdoor belief
            white_risk = self.white_belief[r, c]
            # Black trapdoor could also be here (if there's any residual belief)
            black_risk = self.black_belief[r, c]
        else:
            # Black square - risk from black trapdoor belief
            black_risk = self.black_belief[r, c]
            # White trapdoor could also be here
            white_risk = self.white_belief[r, c]
        
        # Combine risks: if either trapdoor is here, we step on it
        # P(trap) = P(white trap) + P(black trap) - P(both) ≈ P(white) + P(black) for small probs
        combined_risk = white_risk + black_risk - (white_risk * black_risk)
        
        return min(1.0, combined_risk)
    
    def mark_confirmed_trap(self, loc: Tuple[int, int]):
        """Mark a location as containing a confirmed trapdoor."""
        r, c = loc
        parity = (r + c) % 2
        
        if parity == 0:
            self.confirmed_white_trap = loc
            # Zero out belief for white trapdoor
            self.white_belief[:] = 0.0
            self.white_belief[r, c] = 1.0
        else:
            self.confirmed_black_trap = loc
            # Zero out belief for black trapdoor
            self.black_belief[:] = 0.0
            self.black_belief[r, c] = 1.0
    
    @property
    def confirmed_trap_squares(self):
        """Returns set of confirmed trap squares (for backward compatibility)."""
        traps = set()
        if self.confirmed_white_trap:
            traps.add(self.confirmed_white_trap)
        if self.confirmed_black_trap:
            traps.add(self.confirmed_black_trap)
        return traps



class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.tracker = TrapdoorTracker()
        
        # STRATEGY: Corner-first + safe expansion
        # Phase-based gameplay
        self.PHASE_EARLY_END = 30  # Turns left threshold for early game
        self.PHASE_MID_END = 15    # Turns left threshold for mid game
        
        # CRITICAL WEIGHT RETUNING to match wc's winning strategy
        self.W_EGG_DIFF = 8000.0  # WAS 5000 - Eggs are THE priority
        self.W_CENTER_SEEK = 20000.0  # WAS 5000 - MASSIVELY increased to force center control
        self.W_EXPLORATION = 500.0    # WAS 120 - Increased to encourage early expansion
        self.W_TERRITORY = 150.0      # Slightly reduced, eggs matter more
        # Reduce extreme risk aversion so agent will engage opponent
        self.W_RISK = 800.0  # lowered from 50000 to allow more aggressive plays
        self.W_BLOCK_WIN = 50000.0    

        # NEW: OFFENSIVE PRESSURE WEIGHTS
        self.W_INVADE_ENEMY_TERRITORY = 15000.0  # Bonus for being in enemy half
        self.W_DENY_ENEMY_EGG_SQUARES = 20000.0  # Bonus for blocking enemy parity squares
        self.W_PRESSURE_ENEMY_SPAWN = 12000.0   # Bonus for being near enemy spawn
        self.W_AGGRESSIVE_TURD = 18000.0        # Bonus for turds that restrict enemy
        self.W_CORNER_DENIAL = 25000.0          # MASSIVE bonus for denying enemy corners

        self.W_DIAGONAL_CONTROL = 300.0 
        self.W_RETURN_PENALTY = 800.0
        self.W_OSCILLATION_PENALTY = 30000.0
        self.W_CENTRAL_TURD = 650.0    
        self.CORNER_EGG_BONUS = 25000.0  # MASSIVE bonus for corner eggs (3 pts vs 1 pt)

        self.W_TURD_COUNT_DIFF_BASE = 250.0
        self.W_EDGE_TURD_PENALTY = 150.0
        self.W_ADJACENT_TURD_PENALTY = 600.0
        
        # NEW: Endgame egg urgency (wc's secret weapon)
        self.W_EGG_OPPORTUNITY = 5000.0  # Penalty for few available egg squares
        self.W_PATH_TO_EGG = 1000.0     # Bonus for moving toward egg squares
        self.W_ENDGAME_RISK_TOLERANCE = 0.3  # Reduce risk weight in endgame (0.3x normal)
        self.CENTER_AREA_BONUS = 3000.0  # Bonus for being/moving in center 2-5, 2-5
        
        # TURD STRATEGY - Turds are secondary to eggs (only in center)
        self.W_TURD_TRAP_ENEMY_SPAWN = 15000.0  # WAS 60000 - Reduced, eggs first
        self.W_TURD_BLOCK_WINNING_PATH = 12000.0  # WAS 45000 - Reduced
        self.W_TURD_REDUCE_TERRITORY = 1500.0    # WAS 6000 - Reduced
        
        # REVISITING - BALANCED
        # Reduce revisit penalties (avoid completely blocking strategic revisits)
        self.W_REVISIT_EGGED = 30000.0
        self.W_REVISIT_POSITION = 5000.0
        
        # CORNER/EDGE HANDLING
        # Soften corner/edge penalties to allow tactical blocking
        self.W_CORNER_LINGER_PENALTY = 20000.0
        self.W_CORNER_BASE_PENALTY = 1000.0
        self.W_EDGE_PENALTY = 3000.0
        self.W_CORNER_PROXIMITY_PENALTY = 1500.0

        # Territory Control
        self.TERRITORY_MIN_THRESHOLD = 18
        self.W_TERRITORY_PENALTY = 3000.0
        
        # Trapdoor Safety - MORE CONSERVATIVE
        # Allow non-zero trapdoor risk tolerance so agent can engage rather than always avoid
        self.TRAPDOOR_RISK_THRESHOLD = 0.08
        self.CONFIRMED_TRAPDOOR_AVOIDANCE = 0.5  # Cells to avoid around confirmed trapdoors
        
        # Oscillation Detection
        self.position_history = []
        self.HISTORY_LENGTH = 8  # Shorter history
        self.visited_squares = set()
        self.egged_squares_visited = set()
        self.territory_history = []
        self.square_visit_count = defaultdict(int)
        
        # NEW: Exploration tracking
        self.exploration_map = np.zeros((8, 8))  # Track visit counts
        self.last_exploration_direction = None  # Maintain exploration momentum
        
        # State Tracking
        self.last_loc = None 
        self.two_back_loc = None 
        self.last_trapdoor_hit = None  # Track when we hit a trapdoor
        
        # Search controls
        self.start_time = 0
        self.time_limit = 0
        self.my_parity = 0
        self.opp_parity = 0
        self.N_TOTAL_TURNS = 40
        
        # Cache
        self.territory_cache = {}
        self.cache_generation = 0
        # Per-turn caches to avoid repeated expensive computations
        self._per_turn_flood_cache = {}
        # Tunable convenience constants
        self.ADJACENCY_RELAX_TURNS = 3
        self.NEAREST_TURD_SEARCH_RADIUS = 6
        # Turd value thresholds for strategic placement
        self.TURD_VALUE_MIN = 20000.0
        self.TURD_VALUE_PRESSURE_MIN = 3000.0
        # Wall-breaking detection and strategy
        self.WALL_DETECTION_THRESHOLD = 4  # Min opponent turds/eggs to trigger wall-break mode
        self.WALL_BREAK_CORRIDOR_WIDTH = 3  # Max gap width we can push through

        # Logging (lightweight CSV for telemetry when you run matches locally)
        try:
            self.log_path = os.path.join(os.getcwd(), 'agent_play_log.csv')
            if not os.path.exists(self.log_path):
                with open(self.log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp','turns_left','my_loc','opp_loc','action_dir','action_type','my_turds_left','my_eggs_count','opp_eggs_count','oscillation'])
        except Exception:
            # Best-effort logging; don't crash agent if filesystem unavailable
            self.log_path = None

    def _is_on_diagonal(self, loc: Tuple[int, int], map_size: int = 8) -> bool:
        x, y = loc
        return x == y or x + y == map_size - 1

    def _is_on_edge(self, loc: Tuple[int, int], map_size: int = 8) -> bool:
        x, y = loc
        return x == 0 or x == map_size - 1 or y == 0 or y == map_size - 1

    def _is_corner(self, loc: Tuple[int, int], map_size: int = 8) -> bool:
        x, y = loc
        return (x == 0 or x == map_size - 1) and (y == 0 or y == map_size - 1)

    def _get_unexplored_corners(self) -> List[Tuple[int, int]]:
        """Get corners we haven't explored yet with our parity."""
        corners = [(0,0), (0,7), (7,0), (7,7)]
        unexplored = []
        for corner in corners:
            if (corner[0] + corner[1]) % 2 == self.my_parity:
                if corner not in self.visited_squares:
                    unexplored.append(corner)
        return unexplored

    def _get_our_parity_corners(self) -> List[Tuple[int, int]]:
        """Get all corners we can lay eggs on (matching parity)."""
        corners = [(0,0), (0,7), (7,0), (7,7)]
        our_corners = [c for c in corners if (c[0] + c[1]) % 2 == self.my_parity]
        return our_corners

    def _get_unexplored_regions(self) -> List[Tuple[int, int]]:
        """Identify unexplored regions of the board."""
        unexplored = []
        for r in range(8):
            for c in range(8):
                loc = (r, c)
                if loc not in self.visited_squares:
                    # Prioritize squares of our parity
                    if (r + c) % 2 == self.my_parity:
                        unexplored.append(loc)
        return unexplored

    def _comprehensive_move_score(self, move: Tuple[Direction, MoveType], current_loc: Tuple[int, int], state: Dict, turns_left: int) -> float:
        """
        Unified move scoring that considers multiple objectives without early elimination.
        Returns a single score that ranks all moves fairly.
        This promotes BREADTH of search over narrow optimization.
        """
        score = 0.0
        new_loc = loc_after_direction(current_loc, move[0])
        move_type = move[1]
        
        # ===== MUST-TAKE MOVES (very high priority) =====
        # Laying eggs on your parity is critical - do it when possible
        if move_type == MoveType.EGG:
            cell_parity = (current_loc[0] + current_loc[1]) % 2
            if cell_parity == self.my_parity and current_loc not in state['my_eggs']:
                score += 50000.0  # Mandatory egg placement
        
        # ===== STRATEGIC OBJECTIVES (medium-high priority) =====
        # 1. TERRITORY EXPANSION (critical early/mid-game)
        try:
            my_space = self._flood_fill_count(state, True)
            temp_state = self._apply_move_sim(state, move, True)
            if temp_state and self._is_sim_state_valid(temp_state):
                new_space = self._flood_fill_count(temp_state, True)
                delta_space = new_space - my_space
                # Heavier weight on space expansion; amplify late-game urgency
                space_score = delta_space * (800.0 if turns_left <= 15 else 600.0 if turns_left <= 25 else 400.0)
                score += space_score
        except:
            pass
        
        # 2. BLOCKING ENEMY SPACE (reduce opponent's available moves)
        try:
            temp_state = self._apply_move_sim(state, move, True)
            if temp_state and self._is_sim_state_valid(temp_state):
                opp_space_before = self._flood_fill_count(state, False)
                opp_space_after = self._flood_fill_count(temp_state, False)
                blocked = opp_space_before - opp_space_after
                score += blocked * 200.0
        except:
            pass
        
        # 3. WALL BUILDING (connect turds into solid structures)
        if move_type == MoveType.TURD:
            is_wall = self._is_contiguous_wall(new_loc, state)
            if is_wall:
                hor_len = self._count_wall_length(new_loc, state, 'horizontal')
                ver_len = self._count_wall_length(new_loc, state, 'vertical')
                wall_len = max(hor_len, ver_len)
                score += wall_len * 8000.0  # Strong incentive for walls
            else:
                score -= 5000.0  # Penalize isolated turds
        
        # 4. EXPLORATION & AVOIDING OSCILLATION
        moves_from_here = 0
        for dir in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            test_loc = loc_after_direction(new_loc, dir)
            if 0 <= test_loc[0] < 8 and 0 <= test_loc[1] < 8:
                moves_from_here += 1
        
        # Bonus for moving to high-mobility squares (center of board)
        center_dist = manhattan_distance(new_loc, (3.5, 3.5))
        center_bonus = (7.0 - center_dist) * 100.0 if center_dist <= 7 else 0.0
        score += center_bonus
        
        # Bonus for fresh/unvisited squares
        visit_count = self.square_visit_count.get(new_loc, 0)
        if visit_count == 0:
            score += 2000.0  # New square bonus
        elif visit_count == 1:
            score += 500.0   # Recently visited but not much
        else:
            score -= min(visit_count * 200.0, 3000.0)  # Penalty for over-visiting
        
        # 5. BLOCKING OPPONENT ESCAPE ROUTES (reduce opponent immediate options)
        opp_escapes = self._get_opponent_escape_paths(state['opp_loc'], state)
        escapes_blocked = 0
        for esc in opp_escapes:
            if manhattan_distance(new_loc, esc) <= 1:
                escapes_blocked += 1
        score += escapes_blocked * 3000.0
        
        # 6. LATE-GAME EGG GATHERING (when turns are limited, MAXIMIZE egg-laying!)
        if turns_left <= 15 and (new_loc[0] + new_loc[1]) % 2 == self.my_parity:
            if new_loc not in state['my_eggs']:
                # AGGRESSIVE: In endgame, reaching egg squares is CRITICAL
                egg_bonus = 15000.0 if turns_left <= 10 else 8000.0
                score += egg_bonus
        
        # 7. CORNER CONTROL (corners are worth 3 eggs each)
        corners = [(0,0), (0,7), (7,0), (7,7)]
        for corner in corners:
            if (corner[0] + corner[1]) % 2 == self.my_parity and corner not in state['my_eggs']:
                dist_to_corner = manhattan_distance(new_loc, corner)
                corner_score = (4.0 - dist_to_corner) * 1000.0 if dist_to_corner <= 3 else 0.0
                score += corner_score
        
        # 8. AVOID TRAPS (skip if this location looks dangerous)
        if new_loc in self.tracker.confirmed_trap_squares:
            score -= 100000.0  # Never go to confirmed traps
        
        return score
    
    def _choose_move_by_breadth(self, valid_moves: List[Tuple[Direction, MoveType]], current_loc: Tuple[int, int], state: Dict, turns_left: int) -> Tuple[Direction, MoveType]:
        """
        Choose move by scoring all valid moves and picking the highest.
        This promotes breadth (evaluating many moves) vs depth (deeply analyzing few moves).
        """
        if not valid_moves:
            return valid_moves[0] if valid_moves else (Direction.UP, MoveType.PLAIN)
        
        scored_moves = []
        for move in valid_moves:
            score = self._comprehensive_move_score(move, current_loc, state, turns_left)
            scored_moves.append((score, move))
        
        # Sort by score descending
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Pick top-scoring move
        best_score, best_move = scored_moves[0]
        return best_move
        """Detect harmful oscillation - catch tight loops and turd-induced bouncing."""
        if len(self.position_history) < 3:
            return False
        
        # Catch immediate back-and-forth (A-B-A)
        if len(self.position_history) >= 2:
            if new_loc == self.position_history[-2]:
                return True
        
        # Catch tight 2-3 position loop
        if len(self.position_history) >= 4:
            recent = self.position_history[-4:]
            if len(set(recent)) <= 2:  # Only 2 unique positions in last 4 moves
                return True
        
        # NEW: Detect turd-induced oscillation (bouncing between valid moves due to blocked paths)
        # If visiting very few unique squares in recent window
        if len(self.position_history) >= 6:
            recent_6 = self.position_history[-6:]
            unique_recent = len(set(recent_6))
            
            if unique_recent <= 3:
                # Only visiting 3 squares in last 6 moves - likely trapped by turds
                # Check if we're bouncing (visiting same squares repeatedly)
                if recent_6[-1] in recent_6[-4:-1]:  # Last move revisits something from 2-4 moves ago
                    return True
        
        return False

    def _force_exploration_breakout(self, current_loc: Tuple[int, int], valid_moves: List[Tuple[Direction, MoveType]], 
                                    state: Dict, turns_left: int) -> Optional[Tuple[Direction, MoveType]]:
        """
        FORCED EXPLORATION OVERRIDE: When oscillating in late game, force a breakout move.
        
        Strategy:
        1. Pick a direction that maximizes distance from position_history
        2. Prefer unvisited or rarely-visited squares
        3. If stuck in visited territory, move AWAY from recent positions
        
        This is a standalone decision that overrides normal scoring when oscillation is detected.
        """
        if turns_left > 20 or not valid_moves:
            return None  # Only engage in late endgame
        
        best_move = None
        best_score = float('-inf')
        
        recent_positions = set(self.position_history[-6:]) if self.position_history else set()
        
        for move in valid_moves:
            direction, move_type = move
            new_loc = loc_after_direction(current_loc, direction)
            
            if new_loc not in state.get('opp_eggs', set()) and new_loc not in self.tracker.confirmed_trap_squares:
                # SCORE 1: Distance from recent positions (prioritize escape)
                min_dist_to_recent = min(
                    [manhattan_distance(new_loc, pos) for pos in recent_positions]
                ) if recent_positions else 0
                escape_score = min_dist_to_recent * 100.0
                
                # SCORE 2: Visit count (prefer unvisited squares)
                visit_count = self.square_visit_count.get(new_loc, 0)
                exploration_score = 500.0 / (1.0 + visit_count)  # Heavily penalize revisits
                
                # SCORE 3: Prefer egg moves (break out by laying eggs)
                egg_bonus = 1000.0 if move_type == MoveType.EGG else 0.0
                
                # SCORE 4: Direction momentum (prefer continuing in same direction)
                momentum_bonus = 50.0 if direction == self.last_exploration_direction else 0.0
                
                total_score = escape_score + exploration_score + egg_bonus + momentum_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_move = move
        
        print(f"OSCILLATION DETECTED: Forcing exploration breakout. Best move: {best_move}, Score: {best_score:.1f}")
        return best_move

    def _force_corner_escape(self, current_loc: Tuple[int,int], valid_moves: List[Tuple[Direction, MoveType]], state: Dict) -> Optional[Tuple[Direction, MoveType]]:
        """Force an escape move when stuck in a corner or edge area.

        Chooses a move that increases distance to the corner (toward center)
        and preferably reduces opponent mobility. Returns None if no suitable move.
        """
        if not self._is_corner(current_loc) and not self._is_on_edge(current_loc):
            return None

        best = None
        best_score = float('-inf')
        center = (3.5, 3.5)
        for m in valid_moves:
            d, mtype = m
            new_loc = loc_after_direction(current_loc, d)
            # Skip confirmed traps or opponent egg/turd squares
            if new_loc in self.tracker.confirmed_trap_squares:
                continue
            if new_loc in state.get('opp_eggs', set()) or new_loc in state.get('opp_turds', set()):
                continue

            # Score by distance to center (prefer larger decrease)
            cur_dist = manhattan_distance(current_loc, center)
            new_dist = manhattan_distance(new_loc, center)
            center_delta = cur_dist - new_dist

            # Prefer moves that increase distance from the corner (center_delta>0)
            score = center_delta * 500.0

            # Small bonus for plain moves (movement) and egg to allow escape
            if mtype == MoveType.PLAIN:
                score += 50.0
            if mtype == MoveType.EGG:
                score += 200.0

            # If opponent is near, reward moves that reduce their mobility
            try:
                if state and manhattan_distance(state.get('opp_loc'), current_loc) <= 4:
                    opp_before = len(self._get_valid_moves_sim(state, False))
                    ns = self._apply_move_sim(state, m, True)
                    if ns and self._is_sim_state_valid(ns):
                        opp_after = len(self._get_valid_moves_sim(ns, False))
                        score += (opp_before - opp_after) * 300.0
            except Exception:
                pass

            if score > best_score:
                best_score = score
                best = m

        # Only force escape if it's meaningfully better
        if best_score > 0:
            return best
        return None

    def _is_near_confirmed_trapdoor(self, loc: Tuple[int, int]) -> bool:
        """Check if location is dangerously close to confirmed trapdoor."""
        for trap in self.tracker.confirmed_trap_squares:
            if manhattan_distance(loc, trap) <= self.CONFIRMED_TRAPDOOR_AVOIDANCE:
                return True
        return False

    def _should_place_strategic_turd(self, current_loc: Tuple[int, int], state: Dict) -> bool:
        """Determine if we should place a strategic turd at current location."""
        if state.get('my_turds_left', 0) <= 0:
            return False
        
        opp_loc = state['opp_loc']
        opp_spawn = state.get('opp_spawn')
        
        # ONLY place turds in these strategic situations:
        
        # 1. Enemy is at their spawn and we can trap them there
        if opp_spawn and opp_loc == opp_spawn:
            if manhattan_distance(current_loc, opp_spawn) <= 2:
                return True
        
        # 2. Enemy is in a corner and we can reduce their escape routes
        if self._is_corner(opp_loc):
            if manhattan_distance(current_loc, opp_loc) <= 2:
                return True
        
        # 3. We're blocking a path to a high-value corner egg
        corners = [(0,0), (0,7), (7,0), (7,7)]
        for corner in corners:
            if (corner[0] + corner[1]) % 2 == self.opp_parity:
                if corner not in state['opp_eggs']:
                    # This corner is valuable to opponent
                    if manhattan_distance(current_loc, corner) == 1:
                        if manhattan_distance(opp_loc, corner) <= 3:
                            return True
        
        return False

    def _filter_bad_moves(self, moves: List[Tuple[Direction, MoveType]], current_loc: Tuple[int, int]) -> List[Tuple[Direction, MoveType]]:
        """Filter moves - PERMISSIVE to allow late-game expansion."""
        good_moves = []
        bad_moves = []
        
        for move in moves:
            d, m_type = move
            new_loc = loc_after_direction(current_loc, d)
            
            # CRITICAL: Avoid confirmed trapdoors
            if new_loc in self.tracker.confirmed_trap_squares:
                bad_moves.append(move)
                continue
            
            # Allow corner eggs ONLY if corner isn't already egged
            if self._is_corner(new_loc) and m_type == MoveType.EGG:
                if new_loc not in self.egged_squares_visited:
                    good_moves.append(move)
                    continue
                else:
                    # Already egged this corner - treat as revisiting egged square
                    bad_moves.append(move)
                    continue
            
            # Skip corners for non-egg moves ONLY if we're not trapped
            if self._is_corner(new_loc) and m_type != MoveType.EGG:
                if not self._is_corner(current_loc):
                    bad_moves.append(move)
                    continue
            
            # RELAXED: Allow revisiting egged squares - they may be transit routes
            # Only mark as "bad" but still include if no good moves exist
            if new_loc in self.egged_squares_visited:
                bad_moves.append(move)
                continue
            
            # Only prevent tight immediate back-and-forth (position[-1] == new_loc)
            # But allow it if we've been there longer ago
            if len(self.position_history) >= 1 and new_loc == self.position_history[-1]:
                bad_moves.append(move)
                continue
            
            good_moves.append(move)
        
        # PERMISSIVE: Return good moves, but fallback to bad moves (egged squares, etc)
        # This allows traversing previously egged territory to reach new areas
        return good_moves if good_moves else (bad_moves if bad_moves else moves)

    def _score_move_for_exploration(self, move: Tuple[Direction, MoveType], current_loc: Tuple[int, int], state: Dict = None) -> float:
        """AGGRESSIVE scoring - prioritize offense and enemy territory invasion."""
        d, m_type = move
        new_loc = loc_after_direction(current_loc, d)
        score = 0.0
        
        # Trapdoor check (less strict)
        if new_loc in self.tracker.confirmed_trap_squares:
            return -1000000.0
        
        risk = self.tracker.get_risk(new_loc)
        if risk > 0.25:  # Only avoid very high risk
            score -= risk * 20000  # REDUCED penalty
        
        # CRITICAL: Massive penalty for revisiting egged squares (prevents oscillation)
        if new_loc in self.egged_squares_visited:
            score -= 500000.0
        
        # OFFENSIVE PRIORITY 1: INVADE ENEMY TERRITORY
        if state:
            invasion_bonus = self._get_enemy_territory_score(new_loc, state)
            score += invasion_bonus
            
            # Extra bonus for moving TOWARD enemy spawn
            opp_spawn = state.get('opp_spawn')
            if opp_spawn:
                new_dist_to_enemy = manhattan_distance(new_loc, opp_spawn)
                current_dist_to_enemy = manhattan_distance(current_loc, opp_spawn)
                if new_dist_to_enemy < current_dist_to_enemy:
                    # Moving closer to enemy spawn
                    score += self.W_PRESSURE_ENEMY_SPAWN * (current_dist_to_enemy - new_dist_to_enemy)
            
            # CORNER DENIAL
            corner_denial = self._get_corner_denial_value(new_loc, state)
            score += corner_denial
        
        # Exploration bonus (but less important than offense)
        visit_count = self.exploration_map[new_loc[0], new_loc[1]]
        score -= visit_count * 5000  # REDUCED
        
        if new_loc not in self.visited_squares:
            score += 30000  # REDUCED from 100k
            
            if (new_loc[0] + new_loc[1]) % 2 == self.my_parity:
                score += 20000
        
        # Egg moves - ALWAYS valuable
        if m_type == MoveType.EGG:
            if (current_loc[0] + current_loc[1]) % 2 == self.my_parity:
                if current_loc not in self.egged_squares_visited:
                    score += 80000  # MASSIVE egg priority
                    
                    if self._is_corner(current_loc):
                        score += self.CORNER_EGG_BONUS
        
        # AGGRESSIVE TURD PLACEMENT
        if m_type == MoveType.TURD and state:
            # Calculate denial value
            denied_squares = self._count_denied_enemy_egg_squares(current_loc, state)
            score += denied_squares * self.W_DENY_ENEMY_EGG_SQUARES
            
            # Corner denial
            score += self._get_corner_denial_value(current_loc, state)
            
            # Standard turd value
            try:
                turd_value = self._calculate_turd_value(current_loc, state)
                score += turd_value / 2.0
            except:
                pass
        
        # ALLOW edges for transit (reduced penalty)
        if self._is_corner(new_loc) and m_type != MoveType.EGG:
            score -= 5000  # REDUCED
        elif self._is_on_edge(new_loc):
            score -= 1000  # MINIMAL penalty - edges are valid routes

        # OPPONENT INTERCEPT / BLOCKING HEURISTIC
        # If opponent is nearby, prefer moves that move toward them or reduce their mobility
        try:
            if state:
                opp_loc = state.get('opp_loc')
                if opp_loc is not None:
                    cur_dist = manhattan_distance(current_loc, opp_loc)
                    new_dist = manhattan_distance(new_loc, opp_loc)
                    # Bonus for moving closer to opponent (encourage blocking)
                    if new_dist < cur_dist:
                        score += 1200 * (cur_dist - new_dist)

                    # If opponent is relatively close, simulate move and check opponent mobility
                    if cur_dist <= 4:
                        try:
                            opp_moves_before = len(self._get_valid_moves_sim(state, False))
                            ns = self._apply_move_sim(state, move, True)
                            if ns is not None and self._is_sim_state_valid(ns):
                                opp_moves_after = len(self._get_valid_moves_sim(ns, False))
                                delta = opp_moves_before - opp_moves_after
                                if delta > 0:
                                    # Reward moves that reduce opponent mobility
                                    score += delta * 3000
                        except Exception:
                            pass
        except Exception:
            pass

        return score

    def play(self, board_obj: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        self.start_time = time()
        
        self.my_parity = board_obj.chicken_player.even_chicken
        self.opp_parity = board_obj.chicken_enemy.even_chicken

        current_loc = board_obj.chicken_player.get_location()
        spawn = getattr(board_obj.chicken_player, 'starting_square', None)

        self.two_back_loc = self.last_loc
        self.last_loc = current_loc
        
        # Improved trapdoor detection and memory
        if self.last_loc is not None and self.two_back_loc is not None:
            # If we suddenly returned to spawn (teleported via trapdoor)
            if current_loc == spawn:
                # We were teleported back to spawn from the last_loc
                if self.last_loc != spawn:
                    # last_loc must have been the trapdoor
                    possible_trap = self.last_loc
                    
                    # Safety check
                    if possible_trap != spawn:
                        self.tracker.mark_confirmed_trap(possible_trap)
                        print(f"TRAPDOOR DETECTED at {possible_trap}")
            else:
                # We did NOT teleport, so we safely moved from last_loc to current_loc
                # Update Bayesian beliefs with sensor data
                self.tracker.update(current_loc, sensor_data)
        
        # Extract State
        state = self._extract_state(board_obj)
        if state is None:
            state = self._safe_state(board_obj)

        # Clear per-turn caches (so cached flood-fill reflects current state)
        try:
            self._per_turn_flood_cache.clear()
        except Exception:
            self._per_turn_flood_cache = {}
        
        # CRITICAL: Update egged_squares_visited to include all current eggs
        # This prevents oscillation back to already-egged regions
        self.egged_squares_visited.update(state['my_eggs'])
        
        # ADAPTIVE TIME BUDGET - allocate more time when decision is critical
        total_time_rem = time_left()
        turns_rem = board_obj.turns_left_player
        
        # Dynamic budget based on game criticality
        egg_diff = len(state['my_eggs']) - len(state['opp_eggs'])
        is_close_race = abs(egg_diff) <= 2  # Game is competitive
        enemy_distance = manhattan_distance(state['my_loc'], state['opp_loc'])
        is_enemy_near = enemy_distance <= 3
        
        # AGGRESSIVE BUDGET: Use more time for deeper analysis
        if is_close_race and is_enemy_near and turns_rem <= 15:
            # CRITICAL DECISION - allocate MAXIMUM time for endgame
            base_budget = 9.0  # INCREASED from 7.5
        elif turns_rem > 30:
            base_budget = 4.0  # Early game, still use reasonable time
        elif turns_rem > 15:
            base_budget = 7.0  # Mid game - use significant time
        else:
            # Late game - always use maximum
            base_budget = 8.5 if is_close_race else 7.5
        
        safety_buffer = 0.5  # REDUCED from 1.0 to use more time
        calculated_budget = min(base_budget, total_time_rem - safety_buffer)
        self.turn_budget = max(1.0, calculated_budget)
        self.time_limit = self.start_time + self.turn_budget

        # Clear cache
        self.cache_generation += 1
        if self.cache_generation > 5:
            self.territory_cache.clear()
            self.cache_generation = 0

        valid_moves = board_obj.get_valid_moves()
        if not valid_moves:
            return self._emergency_escape_move(current_loc)
        
        # === ULTRA-CRITICAL: PREFER EGG MOVES ABOVE ALL ELSE ===
        # Filter for EGG moves first
        egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
        if egg_moves:
            # We have the option to lay an egg - DO IT
            best_move = egg_moves[0]  # Any egg move is fine; direction doesn't matter
            new_loc = loc_after_direction(current_loc, best_move[0])
            self.position_history.append(new_loc)
            if len(self.position_history) > self.HISTORY_LENGTH:
                self.position_history.pop(0)
            self.square_visit_count[new_loc] += 1
            self.visited_squares.add(new_loc)
            self.last_exploration_direction = best_move[0]
            return best_move
        
        # No egg move available; proceed to normal decision logic
        
        # === ADAPTIVE SEARCH-BASED DECISION ===
        # Use minimax with game-aware depth selection for aggressive play
        turns_left = board_obj.turns_left_player
        dist = manhattan_distance(state['my_loc'], state['opp_loc'])
        
        # MODERATE DEPTH SETTINGS (balance quality vs speed)
        if turns_left <= 10:
            search_depth = 12  # Endgame
        elif turns_left <= 15:
            search_depth = 11  # Critical endgame
        elif turns_left <= 20:
            search_depth = 10  # Late-game
        elif dist <= 2 or turns_left <= 25:
            search_depth = 9  # Enemy nearby or mid-late game
        else:
            search_depth = 8  # Default
        
        # Use minimax for tight tactical decisions
        if dist <= 3 or turns_left <= 25:
            val, best_move = self.minimax(state, search_depth, float('-inf'), float('inf'), True)
            if best_move is None:
                # Fallback to MCTS
                best_move = self.mcts_search(state, time_left)
        else:
            # Early game far from opponent: use MCTS for flexibility
            best_move = self.mcts_search(state, time_left)
        
        # If still no move, use emergency fallback
        if best_move is None:
            best_move = self._choose_move_by_breadth(valid_moves, current_loc, state, turns_left)

        # CRITICAL: Validate best_move before using it
        if best_move is None or (not board_obj.is_valid_move(best_move[0], best_move[1], False)):
            legal = board_obj.get_valid_moves()
            if legal:
                legal = self._filter_bad_moves(legal, current_loc)
                if legal:
                    legal = sorted(legal, key=lambda m: self._score_move_for_exploration(m, current_loc, state), reverse=True)
                    best_move = legal[0]
                else:
                    legal = board_obj.get_valid_moves()
                    best_move = sorted(legal, key=lambda m: self._score_move_for_exploration(m, current_loc, state), reverse=True)[0]
            else:
                return (Direction.UP, MoveType.PLAIN)

        new_loc = loc_after_direction(current_loc, best_move[0])

        if new_loc in self.tracker.confirmed_trap_squares:
            # Force alternative move!
            for alt in valid_moves:
                alt_loc = loc_after_direction(current_loc, alt[0])
                if alt_loc not in self.tracker.confirmed_trap_squares:
                    best_move = alt
                    new_loc = alt_loc
                    break
        
        # Update position history FIRST (before checking oscillation)
        self.position_history.append(new_loc)
        if len(self.position_history) > self.HISTORY_LENGTH:
            self.position_history.pop(0)
        
        # Track visit counts for exploration tracking
        self.square_visit_count[new_loc] += 1
        self.visited_squares.add(new_loc)
        
        # Update exploration momentum
        self.last_exploration_direction = best_move[0]

        print(f"TIME: {time() - self.start_time:.3f}s, Move: {best_move}, Unexplored: {len(self._get_unexplored_regions())}")

        # Telemetry logging: append chosen move and simple state snapshot to CSV
        try:
            if getattr(self, 'log_path', None):
                with open(self.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time(),
                        turns_left,
                        current_loc,
                        state.get('opp_loc'),
                        str(best_move[0]) if best_move else '',
                        getattr(best_move[1], 'name', str(best_move[1])) if best_move else '',
                        state.get('my_turds_left'),
                        len(state.get('my_eggs', [])),
                        len(state.get('opp_eggs', [])),
                        0
                    ])
        except Exception:
            pass

        return best_move

    def _dir_toward_target(self, from_loc: Tuple[int, int], to_loc: Tuple[int, int]) -> Direction:
        """Get direction toward a target location."""
        dx = to_loc[0] - from_loc[0]
        dy = to_loc[1] - from_loc[1]
        
        if abs(dx) > abs(dy):
            return Direction.DOWN if dx > 0 else Direction.UP
        else:
            return Direction.RIGHT if dy > 0 else Direction.LEFT

    def _find_nearest_turd_spot(self, state: Dict, max_search: int = 6) -> Optional[Direction]:
        """Find nearest location we can place a turd (returns first step direction).

        Criteria: spot is empty (no eggs/turds), not a confirmed trap, and either
        - adjacent to one or more opponent-parity empty squares (blocking opportunity), or
        - near enemy spawn or enemy (strategic), or
        - otherwise acceptable if we need to use turds by endgame.
        """
        start = state['my_loc']
        my_eggs = state['my_eggs']
        opp_eggs = state['opp_eggs']
        my_turds = state['my_turds']
        opp_turds = state['opp_turds']
        opp_parity = self.opp_parity

        # BFS to find nearest candidate
        from collections import deque
        q = deque([(start, 0)])
        visited = {start}

        while q:
            loc, dist = q.popleft()
            if dist > max_search:
                break

            # candidate placement is the current loc (we place turd at our current tile)
            if loc not in my_eggs and loc not in my_turds and loc not in opp_eggs and loc not in opp_turds and loc not in self.tracker.confirmed_trap_squares:
                # compute blocking potential
                blocking = 0
                for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:
                    adj = (loc[0] + dx, loc[1] + dy)
                    if 0 <= adj[0] < 8 and 0 <= adj[1] < 8:
                        if adj not in opp_eggs and adj not in my_eggs and adj not in my_turds and adj not in opp_turds and ((adj[0] + adj[1]) % 2) == opp_parity:
                            blocking += 1
                # Favor blocking spots or those near enemy spawn/loc
                if blocking >= 1:
                    # return first step toward this loc
                    if loc == start:
                        return None  # already here (caller will place)
                    return self._dir_toward_target(start, loc)

                # If none blocking, accept near-enemy or near-spawn
                if manhattan_distance(loc, state['opp_loc']) <= 2 or manhattan_distance(loc, state['opp_spawn']) <= 3:
                    if loc == start:
                        return None
                    return self._dir_toward_target(start, loc)

            # expand neighbors
            for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:
                nloc = (loc[0] + dx, loc[1] + dy)
                if 0 <= nloc[0] < 8 and 0 <= nloc[1] < 8 and nloc not in visited:
                    visited.add(nloc)
                    q.append((nloc, dist + 1))

        return None

    def _bfs_first_step_toward(self, start: Tuple[int,int], goal: Tuple[int,int], state: Dict, max_search: int = 20) -> Optional[Direction]:
        """Return the first step Direction toward goal using BFS avoiding opponent obstacles and confirmed traps.
        Uses state's egg/turd sets as obstacles. Returns None if no path found within max_search.
        """
        if start == goal:
            return None
        from collections import deque
        q = deque([(start, [])])
        seen = {start}
        deltas = [(Direction.UP, ( -1, 0 )), (Direction.RIGHT, (0, 1)), (Direction.DOWN, (1, 0)), (Direction.LEFT, (0, -1))]

        opp_eggs = state.get('opp_eggs', set())
        opp_turds = state.get('opp_turds', set())
        my_eggs = state.get('my_eggs', set())
        my_turds = state.get('my_turds', set())
        opp_loc = state.get('opp_loc')

        steps = 0
        while q and steps <= max_search:
            loc, path = q.popleft()
            steps += 1
            for d, (dx, dy) in deltas:
                nloc = (loc[0] + dx, loc[1] + dy)
                if not (0 <= nloc[0] < 8 and 0 <= nloc[1] < 8):
                    continue
                if nloc in seen:
                    continue
                # Can't move into opponent location
                if nloc == opp_loc:
                    continue
                # Avoid confirmed trap squares
                if nloc in self.tracker.confirmed_trap_squares:
                    continue
                # Avoid squares with opponent turds/eggs (we can consider moving adjacent but not onto)
                if nloc in opp_eggs or nloc in opp_turds:
                    continue
                # Avoid our own turds/eggs as stepping onto them is usually non-productive
                if nloc in my_turds or nloc in my_eggs:
                    continue

                new_path = path + [d]
                if nloc == goal:
                    return new_path[0] if new_path else None
                seen.add(nloc)
                q.append((nloc, new_path))

        return None

    # [Include all other methods from original agent - minimax, evaluate, mcts_search, etc.]
    # I'll include the key modified ones:

    def _order_moves(self, moves: List[Tuple[Direction, MoveType]], state: Dict) -> List[Tuple[Direction, MoveType]]:
        """Order moves with improved exploration and trapdoor avoidance."""
        def move_priority(move):
            d, m_type = move
            score = 0
            new_loc = loc_after_direction(state['my_loc'], d)
            
            # Bounds check - invalid move
            if not (0 <= new_loc[0] < 8 and 0 <= new_loc[1] < 8):
                return -1000000
            
            # CRITICAL: Avoid confirmed trapdoors
            if self._is_near_confirmed_trapdoor(new_loc):
                return -1000000
            
            # High trapdoor risk
            risk = self.tracker.get_risk(new_loc)
            if risk > 0.01:
                score -= risk * 50000
            
            # Prioritize egg moves on unexplored our-parity squares
            if m_type == MoveType.EGG:
                score += 6000
                if self._is_corner(state['my_loc']):
                    score += self.CORNER_EGG_BONUS
                if state['my_loc'] not in self.egged_squares_visited:
                    score += 8000
            
            # STRATEGIC TURD ONLY
            elif m_type == MoveType.TURD:
                # Use calculated turd value to prioritize placements
                try:
                    turd_value = self._calculate_turd_value(state['my_loc'], state)
                except Exception:
                    turd_value = 0.0
                # Scale turd_value into ordering score (divisor chosen to keep magnitude reasonable)
                score += max(-50000, min(50000, turd_value / 2.0))
            
            # MASSIVE exploration bonus
            if new_loc not in self.visited_squares:
                score += self.W_EXPLORATION
                if (new_loc[0] + new_loc[1]) % 2 == self.my_parity:
                    score += 30000
            
            # Penalty for revisiting egged squares
            if new_loc in self.egged_squares_visited:
                score -= self.W_REVISIT_EGGED
            
            # Light penalty for tight oscillation
            if len(self.position_history) >= 2 and new_loc in self.position_history[-2:]:
                score -= 20000
            
            return score
        
        return sorted(moves, key=move_priority, reverse=True)

    def minimax(self, state, depth, alpha, beta, maximizing):
        if state is None or depth == 0:
            if state is None:
                return -1e9 if maximizing else 1e9, None
            return self.evaluate(state), None
        
        if time() > self.time_limit:
            raise TimeoutError()

        moves = self._get_valid_moves_sim(state, is_me=maximizing)
        if not moves:
            return self.evaluate_terminal(state, maximizing), None
        
        try:
            enemy_loc = state['opp_loc']
            my_loc = state['my_loc']
            moves = sorted(
                moves,
                key = lambda m: manhattan_distance(loc_after_direction(my_loc, m[0]), enemy_loc)
            )
        except:
            pass

        # Use improved move ordering
        if maximizing:
            moves = self._order_moves_sim(moves, state, True)
        
        best_move = moves[0]
        safe_moves = []
        risks = {}
        
        # Trapdoor pruning
        for move in moves:
            current_loc = state['my_loc'] if maximizing else state['opp_loc']
            next_loc = loc_after_direction(current_loc, move[0])
            
            r, c = next_loc
            # Safety guard!
            if r < 0 or r > 7 or c < 0 or c > 7:
                # Extremely bad state: punish harshly!
                risk = 9999
            else:
                risk = self.tracker.get_risk(next_loc)
            risks[move] = risk
            
            if risk <= self.TRAPDOOR_RISK_THRESHOLD:
                safe_moves.append(move)

        # Filter safe moves to exclude corners/edges/oscillation
        if safe_moves:
            current_loc_for_filtering = state['my_loc'] if maximizing else state['opp_loc']
            safe_moves_filtered = self._filter_bad_moves(safe_moves, current_loc_for_filtering)
            moves_to_search = safe_moves_filtered if safe_moves_filtered else safe_moves
            best_move = moves_to_search[0] if moves_to_search else safe_moves[0]
        else:
            moves_to_search = sorted(moves, key=lambda m: risks[m])[:10]  # Limit to top 10
            best_move = moves_to_search[0]

        if maximizing:
            max_eval = float('-inf')
            for move in moves_to_search:
                new_state = self._apply_move_sim(state, move, is_me=True)
                eval_val, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves_to_search:
                new_state = self._apply_move_sim(state, move, is_me=False)
                eval_val, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def _order_moves_sim(self, moves: List[Tuple[Direction, MoveType]], state: Dict, is_me: bool) -> List[Tuple[Direction, MoveType]]:
        """Fast move ordering for simulation."""
        def priority(move):
            score = 0
            if move[1] == MoveType.EGG:
                score = 10
            elif move[1] == MoveType.TURD:
                score = 8  # Increased turd priority - almost as important as eggs
                # Add strategic value if possible
                try:
                    current_loc = state['my_loc'] if is_me else state['opp_loc']
                    turd_value = self._calculate_turd_value(current_loc, state) / 5000.0
                    score += min(turd_value, 5.0)  # Cap bonus
                except:
                    pass
            return score
        return sorted(moves, key=priority, reverse=True)

    def evaluate(self, state):
        turns_rem = state.get('turns_left_player', 40)
        game_phase = 1.0 - (turns_rem / 40.0)
        
        score = (state['my_score'] - state['opp_score']) * self.W_EGG_DIFF
        
        # Risk (minimal penalty)
        risk = self.tracker.get_risk(state['my_loc'])
        risk_multiplier = 1.0
        if turns_rem < 10:
            risk_multiplier *= self.W_ENDGAME_RISK_TOLERANCE
        score -= risk * self.W_RISK * risk_multiplier
        
        # OFFENSIVE POSITIONING
        current_loc = state['my_loc']
        
        # Invasion bonus
        invasion_score = self._get_enemy_territory_score(current_loc, state)
        score += invasion_score
        
        # Enemy spawn pressure
        opp_spawn = state.get('opp_spawn')
        if opp_spawn:
            dist_to_enemy_spawn = manhattan_distance(current_loc, opp_spawn)
            if dist_to_enemy_spawn <= 4:
                score += self.W_PRESSURE_ENEMY_SPAWN * (5 - dist_to_enemy_spawn)
        
        # Corner denial
        score += self._get_corner_denial_value(current_loc, state)
        
        # Center control (moderate)
        center_dist = manhattan_distance(current_loc, (3.5, 3.5))
        score += self.W_CENTER_SEEK * (7 - center_dist) * 0.8
        
        # Territory (reduced weight)
        cache_key = (frozenset(state['my_eggs']), frozenset(state['opp_eggs']),
                    frozenset(state['my_turds']), frozenset(state['opp_turds']),
                    state['my_loc'], state['opp_loc'])
        
        if cache_key in self.territory_cache:
            my_space, opp_space = self.territory_cache[cache_key]
        else:
            my_space = self._flood_fill_count(state, is_me_perspective=True)
            opp_space = self._flood_fill_count(state, is_me_perspective=False)
            self.territory_cache[cache_key] = (my_space, opp_space)
        
        score += (my_space - opp_space) * self.W_TERRITORY
        
        # AGGRESSIVE TURD EVALUATION
        for turd_loc in state['my_turds']:
            # Each turd should deny enemy squares
            denied = self._count_denied_enemy_egg_squares(turd_loc, state)
            score += denied * self.W_DENY_ENEMY_EGG_SQUARES
            
            # Corner denial value
            score += self._get_corner_denial_value(turd_loc, state)
        
        # Penalize defensive turds in OUR territory
        for turd_loc in state['my_turds']:
            if manhattan_distance(turd_loc, state.get('my_spawn', (0,0))) <= 3:
                score -= 10000  # Penalty for defensive turds
        
        # Corners and edges (minimal penalties - they're transit routes)
        if self._is_corner(current_loc):
            if current_loc in state['my_eggs']:
                score -= 300000  # Force leaving egged corners
            else:
                score -= 5000
        elif self._is_on_edge(current_loc):
            score -= 2000  # Minimal - edges are fine
        
        # Mobility
        mobility = len(self._get_valid_moves_sim(state, True))
        score += mobility * 200
        
        # OFFENSIVE MOBILITY - restrict enemy heavily
        opp_mobility = len(self._get_valid_moves_sim(state, False))
        score -= opp_mobility * 300  # INCREASED - always restrict enemy
        
        # Oscillation prevention
        if current_loc in self.position_history[-4:]:
            score -= 30000
        
        if current_loc in self.egged_squares_visited:
            score -= 200000
        
        return score

    def evaluate_terminal(self, state, am_i_blocked):
        base = self.evaluate(state)
        return base - self.W_BLOCK_WIN if am_i_blocked else base + self.W_BLOCK_WIN

    def _count_available_egg_squares(self, state: Dict, is_me: bool) -> int:
        """Count how many squares are still available for laying eggs."""
        parity = self.my_parity if is_me else self.opp_parity
        count = 0
        
        for x in range(8):
            for y in range(8):
                if (x + y) % 2 != parity:
                    continue
                loc = (x, y)
                if loc in state['my_eggs'] or loc in state['opp_eggs']:
                    continue
                if loc in state['my_turds'] or loc in state['opp_turds']:
                    continue
                count += 1
        
        return count

    def _flood_fill_count(self, state: Dict, is_me_perspective: bool) -> int:
        # Use per-turn caching to avoid repeated expensive flood-fills
        start_node = state['my_loc'] if is_me_perspective else state['opp_loc']
        try:
            key = (
                bool(is_me_perspective),
                start_node,
                frozenset(state.get('my_eggs', set())),
                frozenset(state.get('opp_eggs', set())),
                frozenset(state.get('my_turds', set())),
                frozenset(state.get('opp_turds', set())),
            )
            cached = self._per_turn_flood_cache.get(key)
            if cached is not None:
                return cached
        except Exception:
            key = None
        
        if is_me_perspective:
            obstacles = state['opp_eggs']
            enemy_turds = state['opp_turds']
        else:
            obstacles = state['my_eggs']
            enemy_turds = state['my_turds']
        
        queue = deque([start_node])
        visited = {start_node}
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        while queue:
            curr = queue.popleft()
            
            for dx, dy in deltas:
                nx, ny = curr[0] + dx, curr[1] + dy
                new_loc = (nx, ny)
                
                if not (0 <= nx < 8 and 0 <= ny < 8):
                    continue
                if new_loc in visited:
                    continue
                if new_loc in obstacles:
                    continue
                if new_loc in enemy_turds:
                    continue

                blocked_by_turd = False
                for tdx, tdy in deltas:
                    if (nx + tdx, ny + tdy) in enemy_turds:
                        blocked_by_turd = True
                        break
                if blocked_by_turd:
                    continue
                
                visited.add(new_loc)
                queue.append(new_loc)
        
        result = len(visited) * 0.6 if self._is_corner(start_node) else len(visited)
        try:
            if key is not None:
                self._per_turn_flood_cache[key] = result
        except Exception:
            pass
        return result

    def _extract_state(self, board_obj: Board) -> Dict:
        try:
            my_spawn = getattr(board_obj.chicken_player, 'starting_square', None)
        except Exception:
            my_spawn = None
        try:
            opp_spawn = getattr(board_obj.chicken_enemy, 'starting_square', None)
        except Exception:
            opp_spawn = None

        # Fallback if attribute not present: use their current loc as placeholder!
        if my_spawn is None:
            my_spawn = board_obj.chicken_player.get_location()
        if opp_spawn is None:
            opp_spawn = board_obj.chicken_enemy.get_location()

        return {
            'my_loc': board_obj.chicken_player.get_location(),
            'opp_loc': board_obj.chicken_enemy.get_location(),
            'my_turds_left': board_obj.chicken_player.get_turds_left(),
            'opp_turds_left': board_obj.chicken_enemy.get_turds_left(),
            'my_score': board_obj.chicken_player.get_eggs_laid(),
            'opp_score': board_obj.chicken_enemy.get_eggs_laid(),
            'my_eggs': set(board_obj.eggs_player),
            'opp_eggs': set(board_obj.eggs_enemy),
            'my_turds': set(board_obj.turds_player),
            'opp_turds': set(board_obj.turds_enemy),
            'map_size': 8,
            'turns_left_player': board_obj.turns_left_player,
            'last_loc': self.last_loc,
            'two_back_loc': self.two_back_loc,
            'my_spawn': my_spawn,
            'opp_spawn': opp_spawn,
        }

    def _apply_move_sim(self, state: Dict, move: Tuple[Direction, MoveType], is_me: bool) -> Dict:
        ns = copy.deepcopy(state)
        ns['my_eggs'] = set(ns.get('my_eggs', set()))
        ns['opp_eggs'] = set(ns.get('opp_eggs', set()))
        ns['my_turds'] = set(ns.get('my_turds', set()))
        ns['opp_turds'] = set(ns.get('opp_turds', set()))

        d, m_type = move
        curr_loc = ns['my_loc'] if is_me else ns['opp_loc']
        other_loc = ns['opp_loc'] if is_me else ns['my_loc']

        # If current location is invalid, then abort
        if not (0 <= curr_loc[0] < 8 and 0 <= curr_loc[1] < 8):
            return self._calculate_invalid_state(state, is_me)
            
        # PRECHECK: Cannot drop egg/turd where opponent has egg/turd!
        if m_type == MoveType.EGG:
            if curr_loc in (ns['opp_eggs'] | ns['opp_turds']):
                losing = self.copy_state(state)
                losing['my_score'] -= 10
                return losing
        if m_type == MoveType.TURD:
            if self._flood_fill_count(ns, True) < self._flood_fill_count(state, True):
                return self._calculate_invalid_state(state, is_me)
            if ns.get('my_turds_left', 0) <= 0:
                losing = self.copy_state(state)
                losing['my_score'] -= 10
                return losing
            if curr_loc in (ns['opp_eggs'] | ns['opp_turds']):
                losing = self.copy_state(state)
                losing['my_score'] -= 10
                return losing

        # Apply egg/turd to the current square (egg step/turd step semantics)!
        if m_type == MoveType.EGG:
            if is_me:
                ns['my_eggs'].add(curr_loc)
                ns['my_score'] = ns.get('my_score', 0) + 1
            else:
                ns['opp_eggs'].add(curr_loc)
                ns['opp_score'] = ns.get('opp_score', 0) + 1
        elif m_type == MoveType.TURD:
            if is_me:
                ns['my_turds'].add(curr_loc)
                ns['my_turds_left'] = ns.get('my_turds_left', 0) - 1
            else:
                ns['opp_turds'].add(curr_loc)
                ns['opp_turds_left'] = ns.get('opp_turds_left', 0) - 1

        # Compute movement!
        new_loc = loc_after_direction(curr_loc, d)
        if new_loc in self.tracker.confirmed_trap_squares:
            # INVALID SIM MOVE — punish with extreme loss
            new_state = copy.deepcopy(state)
            new_state['my_score'] -= 99999
            return new_state
        r, c = new_loc
        if r < 0 or r > 7 or c < 0 or c > 7:
            return None
        if not (0 <= new_loc[0] < 8 and 0 <= new_loc[1] < 8):
            return self._calculate_invalid_state(state, is_me)

        # Illegal: cannot move into opponent's position!
        if new_loc == other_loc:
            return self._calculate_invalid_state(state, is_me)
        
        # Illegal: cannot move into square that contains opponent's egg/turd!
        if is_me and (new_loc in (ns['opp_eggs'] | ns['opp_turds'])):
            return self._calculate_invalid_state(state, is_me)
        if (not is_me) and (new_loc in (ns['my_eggs'] | ns['my_turds'])):
            return self._calculate_invalid_state(state, is_me)

        # Assign new position!
        if is_me:
            ns['my_loc'] = new_loc
            ns['two_back_loc'] = ns.get('last_loc')
            ns['last_loc'] = new_loc
            ns['turns_left_player'] = max(0, ns.get('turns_left_player', self.N_TOTAL_TURNS) - 1)
        else:
            ns['opp_loc'] = new_loc

        # Verify invariants!
        if not self._is_sim_state_valid(ns):
            return self._calculate_invalid_state(state, is_me)

        return ns

    def _get_valid_moves_sim(self, state: Dict, is_me: bool) -> List[Tuple[Direction, MoveType]]:
        if state is None:
            return []
        
        moves = []
        curr_loc = state['my_loc'] if is_me else state['opp_loc']
        other_loc = state['opp_loc'] if is_me else state['my_loc']
        turds_left = state['my_turds_left'] if is_me else state['opp_turds_left']
        required_parity = self.my_parity if is_me else self.opp_parity
        cell_parity = (curr_loc[0] + curr_loc[1]) % 2
        my_eggs = state['my_eggs'] if is_me else state['opp_eggs']
        opp_eggs = state['opp_eggs'] if is_me else state['my_eggs']
        my_turds = state['my_turds'] if is_me else state['opp_turds']
        opp_turds = state['opp_turds'] if is_me else state['my_turds']
        dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        is_curr_loc_empty = (curr_loc not in my_eggs) and (curr_loc not in my_turds) and (curr_loc not in opp_eggs) and (curr_loc not in opp_turds)
        can_egg_step = ((cell_parity == required_parity) and is_curr_loc_empty)
        can_lay_egg_here = (curr_loc not in my_eggs) and ((curr_loc[0] + curr_loc[1]) % 2 == required_parity)
        trap_risk = self.tracker.get_risk(curr_loc)

        if trap_risk > 0.05:
            return []
        
        turns_rem = state.get('turns_left_player', 40)

        # CRITICAL: Mid/Late-game turd forcing - if we have turds left and few turns remain, prioritize turd placement
        # (force earlier to ensure all turds get used). Also force when turns remaining <= turds left
        # Be more aggressive: trigger earlier (<=25)
        if is_me and turds_left > 0 and (turns_rem <= 25 or turns_rem <= turds_left) and is_curr_loc_empty:
            # Force turd placement moves
            return [(Direction.UP, MoveType.TURD), (Direction.RIGHT, MoveType.TURD),
                    (Direction.DOWN, MoveType.TURD), (Direction.LEFT, MoveType.TURD)]
        
        if can_lay_egg_here:
            # CRITICAL: Force egg step on correct parity squares - ALWAYS
            # If we're on a square where we can lay eggs, we MUST do so
            # (this gets more points than moving and doing plain)
            return [(Direction.UP, MoveType.EGG), (Direction.RIGHT, MoveType.EGG),
                    (Direction.DOWN, MoveType.EGG), (Direction.LEFT, MoveType.EGG)]
        
        has_adjacent_turd = False
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for dx, dy in deltas:
            adj_loc = (curr_loc[0] + dx, curr_loc[1] + dy)
            if adj_loc in my_turds:
                has_adjacent_turd = True
                break
        
        enemy_loc = state['opp_loc']
        dist_enemy = manhattan_distance(curr_loc, enemy_loc)

        in_enemy_region = manhattan_distance(curr_loc, state['opp_spawn']) <= 4
        in_my_region = manhattan_distance(curr_loc, state['my_spawn']) <= 4

        # STRATEGIC TURD PLACEMENT
        # Turds are valuable for blocking enemy movement and preventing egg placement. Place them:
        # 1. Next to empty opponent-parity squares (block enemy egg opportunities)
        # 2. Near enemy's spawn (trap them)
        # 3. Around enemy's current position (restrict movement)
        # 4. Spread out (not clustered) to maximize coverage
        # 5. DEFENSIVE: If trailing, place turds between enemy spawn and our territory
        turns_rem = state.get('turns_left_player', 40)
        game_phase = (40 - turns_rem) / 40.0  # 0 = early, 1 = late
        
        # Check: is this a strategic location for a turd?
        dist_to_enemy_spawn = manhattan_distance(curr_loc, state['opp_spawn'])
        
        # Determine if we're trailing (need defensive strategy)
        egg_diff = len(my_eggs) - len(opp_eggs)
        is_trailing = egg_diff < -1
        
        # DEFENSIVE POSITIONING - If trailing, build a wall between enemy and our territory
        dist_to_my_spawn = manhattan_distance(curr_loc, state['my_spawn'])
        is_on_defensive_line = (dist_to_my_spawn >= 2 and dist_to_enemy_spawn >= 2 and 
                                dist_to_my_spawn <= dist_to_enemy_spawn)
        
        # BLOCKING EMPTY OPPONENT PARITY SQUARES - Check if we're adjacent to empty squares
        # where opponent can lay eggs
        blocking_value = 0
        for dx, dy in deltas:
            adj_loc = (curr_loc[0] + dx, curr_loc[1] + dy)
            if (0 <= adj_loc[0] < 8 and 0 <= adj_loc[1] < 8):
                # Check if this is an empty square of opponent's parity
                if (adj_loc not in opp_eggs and adj_loc not in my_eggs and
                    adj_loc not in my_turds and adj_loc not in opp_turds and
                    (adj_loc[0] + adj_loc[1]) % 2 == self.opp_parity):
                    # This is a valuable square for opponent to egg on
                    blocking_value += 1
        
        # Be aggressive: block even a single valuable opponent parity square
        has_blocking_opportunity = blocking_value >= 1
        
        # Strategic turd placement criteria:
        can_turd_step = (
            turds_left > 0
            and is_curr_loc_empty
            # Allow adjacency more readily (we'll be aggressive)
            and (not has_adjacent_turd or turns_rem <= max(self.ADJACENCY_RELAX_TURNS, 8))
            and (
                # Priority 1: BLOCK OPPONENT EGG OPPORTUNITIES - Place next to empty opponent squares
                (has_blocking_opportunity and game_phase >= 0.2)
                or
                # Priority 2: Trap at enemy spawn (early game aggression)
                (dist_to_enemy_spawn <= 3 and game_phase <= 0.4)
                or
                # Priority 3: Around enemy current position
                (dist_enemy <= 2)
                or
                # Priority 4: DEFENSIVE - Build wall if trailing
                (is_trailing and is_on_defensive_line)
                or
                # Priority 5: In enemy territory (for map control)
                (in_enemy_region and game_phase >= 0.3 and not is_trailing)
                or
                # Priority 6: Late game, block any nearby territory
                (game_phase >= 0.7 and dist_enemy <= 6)
            )
        )
        
        # Compute turd value at current location to decide whether to allow turd moves
        try:
            turd_value_here = self._calculate_turd_value(curr_loc, state) if turds_left > 0 else 0.0
        except Exception:
            turd_value_here = 0.0
        
        allow_turd_by_value = turd_value_here >= self.TURD_VALUE_MIN
        allow_turd_by_pressure = (turd_value_here >= self.TURD_VALUE_PRESSURE_MIN) and any((curr_loc[0]+dx, curr_loc[1]+dy) in opp_turds for dx,dy in deltas)
        
        for d in dirs:
            new_loc = loc_after_direction(curr_loc, d)
            if new_loc in self.tracker.confirmed_trap_squares:
                continue
            nx, ny = new_loc
            if nx < 0 or nx > 7 or ny < 0 or ny > 7:
                continue
            if new_loc == other_loc:
                continue
            if new_loc in opp_eggs or new_loc in opp_turds:
                continue
            if new_loc in my_eggs or new_loc in my_turds:
                continue
            
            # Allow moving near opponent turds (be willing to fight / counter-lay)
            # Previously we skipped moves adjacent to opponent turds; remove that restriction to swap roles.
                
            # PRIORITY: If we're on correct parity square, ALWAYS add EGG move first
            if can_egg_step and cell_parity == required_parity:
                # Current location is correct parity - egg move has priority
                moves.append((d, MoveType.EGG))
                # Also allow plain movement to escape if needed
                moves.append((d, MoveType.PLAIN))
            else:
                # Normal moves
                moves.append((d, MoveType.PLAIN))
                # Only add egg if we can
                if can_egg_step:
                    moves.append((d, MoveType.EGG))
            
            # Turd moves: include only when strategic (by calculated value) or under pressure
            if can_turd_step and (allow_turd_by_value or has_blocking_opportunity or dist_enemy <= 2):
                moves.append((d, MoveType.TURD))
            else:
                # Consider placement under pressure if turd_value meets lower threshold
                if allow_turd_by_pressure:
                    moves.append((d, MoveType.TURD))
        
        # REMOVED corner special handling - corners should be avoided entirely
        return moves
    
    def mcts_search(self, root_state, time_left):
        root = MCTSNode(copy.deepcopy(root_state))
        start = time()

        # We quickly expand the root here so that children is not empty!
        root_moves = self._get_valid_moves_sim(root.state, True)
        
        # FILTER OUT BAD MOVES before expanding root!
        current_loc_in_state = root.state['my_loc']
        root_moves = self._filter_bad_moves(root_moves, current_loc_in_state)
        
        # If filtering removed all moves, fall back to all moves but with penalties in rollout
        if not root_moves:
            root_moves = self._get_valid_moves_sim(root.state, True)
        
        for m in root_moves:
            child_state = self._apply_move_sim(root.state, m, True)
            if child_state is None:
                continue
            if not self._is_sim_state_valid(child_state):
                continue
            root.children.append(MCTSNode(child_state, parent=root, move=m))

        # We have a fallback in case the root has no moves!
        if not root.children:
            # We return a legal engine move (this is rare)!
            engine_moves = list(self._order_moves(self._get_valid_moves_sim(root_state, True), root_state))
            return engine_moves[0] if engine_moves else (Direction.UP, MoveType.PLAIN)

        # We're dynamically doing rollout based on remaining time!
        # Use 80% of allocated turn budget to ensure we do substantial MCTS
        budget = min(self.turn_budget * 0.80, max(0.1, time_left() - 0.5))
        while time() - start < budget:
            node = self.mcts_select(root)
            reward = self.mcts_rollout(copy.deepcopy(node.state))
            self.mcts_backpropagate(node, reward)

        # We pick the child that has the highest expected score, BUT FILTER OUT BAD MOVES!
        # First, filter out corner/edge/oscillating moves - but allow corner eggs
        good_children = []
        current_loc_in_state = root.state['my_loc']
        for c in root.children:
            new_loc = loc_after_direction(current_loc_in_state, c.move[0])
            
            # CRITICAL: Never step on confirmed trapdoors
            if new_loc in self.tracker.confirmed_trap_squares:
                continue
            
            if self._is_corner(new_loc):
                # Allow corner egg moves, but filter out plain moves to corners
                if c.move[1] == MoveType.EGG:
                    # Corner eggs are valuable - allow them
                    good_children.append(c)
                # Skip plain moves to corners (discourage lingering)
                continue
            if self._is_on_edge(new_loc) and not (self._is_on_edge(current_loc_in_state) or self._is_corner(current_loc_in_state)):
                continue
            if new_loc in self.position_history[-2:]:
                continue
            if new_loc in self.egged_squares_visited:
                continue
            good_children.append(c)
        
        # If we filtered everything, use all children (trapped situation)
        children_to_choose_from = good_children if good_children else root.children
        
        # Prefer egg moves, but turds are also high priority if strategic
        egg_children = [c for c in children_to_choose_from if c.move[1] == MoveType.EGG]
        turd_children = [c for c in children_to_choose_from if c.move[1] == MoveType.TURD]
        
        if egg_children:
            best_child = max(egg_children, key=lambda c: c.total_reward / max(c.visits, 1))
            # Check if a strategic turd is competitive
            if turd_children:
                best_turd = max(turd_children, key=lambda c: c.total_reward / max(c.visits, 1))
                egg_value = best_child.total_reward / max(best_child.visits, 1)
                turd_value = best_turd.total_reward / max(best_turd.visits, 1)
                # If turd is within 20% of egg value, prefer turd (more strategic)
                if turd_value >= egg_value * 0.8:
                    best_child = best_turd
        elif turd_children:
            best_child = max(turd_children, key=lambda c: c.total_reward / max(c.visits, 1))
        else:
            best_child = max(children_to_choose_from, key=lambda c : c.total_reward / max(c.visits, 1))
        return best_child.move
    
    def mcts_select(self, node):
        # select until leaf or expandable node
        while True:
            moves = self._get_valid_moves_sim(node.state, True)
            if not moves:
                return node

            # Expand if there are unexplored moves
            if len(node.children) < len(moves):
                used_moves = {c.move for c in node.children}
                # Prioritize: eggs first, then strategic turds, then others
                def expansion_priority(m):
                    if m[1] == MoveType.EGG:
                        return 0
                    elif m[1] == MoveType.TURD:
                        return 1  # Turds are second priority
                    else:
                        return 2
                egg_first = sorted(moves, key=expansion_priority)
                for m in egg_first:
                    if m in used_moves:
                        continue
                    new_state = self._apply_move_sim(node.state, m, True)
                    if new_state is None or not self._is_sim_state_valid(new_state):
                        continue
                    child = MCTSNode(new_state, parent=node, move=m)
                    node.children.append(child)
                    return child

            # all children expanded; pick by UCB, but break ties randomly and prefer low-visit children slightly
            def ucb_with_jitter(c):
                base_ucb = c.ucb_score()
                # small jitter proportional to 1/sqrt(visits) to prefer less-visited almost-ties
                jitter = random.uniform(0, 1.0 / math.sqrt(max(1, c.visits)))
                return base_ucb + jitter

            node = max(node.children, key=ucb_with_jitter)


    def mcts_rollout(self, rollout_state):
        MAX_SIM_MOVES = 20
        curr_state = copy.deepcopy(rollout_state)

        for _ in range(MAX_SIM_MOVES):
            moves = self._get_valid_moves_sim(curr_state, True)
            if not moves:
                return self.evaluate_terminal(curr_state, True)
            
            # We choose moves weighted by heuristic quality!
            weighted = []
            for m in moves:
                d, mt = m
                new_loc = loc_after_direction(curr_state['my_loc'], d)

                if new_loc in self.tracker.confirmed_trap_squares:
                    continue

                # Safety bounds check!
                r, c = new_loc
                if r < 0 or r >= 8 or c < 0 or c >= 8:
                    # Extremely bad move: punish harshly!
                    return -9999

                risk = self.tracker.get_risk(new_loc)
                
                if risk > self.TRAPDOOR_RISK_THRESHOLD:
                    continue

                # --- beginning of weight calculation (inside the rollout loop) ---
                base = 1.0

                # MASSIVE penalty for stepping onto a tile we've already egged
                if new_loc in curr_state.get('my_eggs', set()):
                    base *= 0.001   # Nearly eliminate this possibility

                # Strong penalty for recently visited positions to encourage exploration
                recent = set(self.position_history[-6:]) if self.position_history else set()
                if new_loc in recent:
                    base *= 0.1  # Much stronger penalty
                
                # Additional penalty if we've visited this square before (from visited_squares)
                if hasattr(self, 'visited_squares') and new_loc in self.visited_squares:
                    base *= 0.3  # Discourage revisiting

                recent_eggs = curr_state.get('_recent_eggs', set())
                if new_loc in recent_eggs:
                    base *= 0.1

                # small epsilon exploration: occasionally ignore heuristics
                if random.random() < 0.05:
                    base = max(base, 0.5) + random.random()

                # STRONG reward for center proximity (encourages leaving edges/corners)
                center = (3.5, 3.5)
                dist_to_center = manhattan_distance(new_loc, center)
                current_dist_to_center = manhattan_distance(curr_state['my_loc'], center)
                # Bonus for moving closer to center, penalty for moving away
                if dist_to_center < current_dist_to_center:
                    base += (current_dist_to_center - dist_to_center) * 10.0  # Very strong reward
                elif dist_to_center > current_dist_to_center:
                    base *= 0.5  # Penalize moving away from center
                base += max(0, 7 - dist_to_center) * 5.0  # Additional center proximity reward
                
                # PENALIZE corners and edges in rollout - but allow corner eggs
                if self._is_corner(new_loc):
                    if mt == MoveType.EGG:
                        base += 25.0  # STRONG bonus for corner eggs (2x score!)
                        # Penalty if we've been to this corner recently (lingering)
                        recent_corners = curr_state.get('_recent_corners', set())
                        if new_loc in recent_corners:
                            base *= 0.3  # Strong penalty for revisiting same corner
                    else:
                        base *= 0.01  # Nearly eliminate plain moves to corners (no lingering)
                elif self._is_on_edge(new_loc):
                    base *= 0.3   # Strong penalty for edges
                
                # EXPLORATION BONUS - MASSIVE reward for visiting new squares
                if hasattr(self, 'visited_squares') and new_loc not in self.visited_squares:
                    base += 50.0  # MASSIVE exploration bonus
                    # Bonus for moving AWAY from our own territory toward free space
                    free_space = self._flood_fill_count(curr_state, True)
                    nearby_own_count = self._count_nearby_own_eggs_turds(curr_state['my_loc'], curr_state, radius=2)
                    if free_space > 10 and nearby_own_count >= 2:
                        # Moving away from our territory toward free space - big bonus
                        base += 30.0
                elif hasattr(self, 'visited_squares') and new_loc in self.visited_squares:
                    base *= 0.2  # Strong penalty for revisiting
                
                # TERRITORIAL CIRCLING PENALTY in rollout
                nearby_own_count = self._count_nearby_own_eggs_turds(new_loc, curr_state, radius=2)
                free_space = self._flood_fill_count(curr_state, True)
                if free_space > 10 and nearby_own_count >= 2:
                    # Moving to area near our territory when free space exists - penalize
                    base *= 0.3  # Strong penalty for territorial circling

                if mt == MoveType.EGG:
                    # keep eggs attractive, but not overwhelmingly so
                    base += 12.0
                    # if current cell was already egged, discourage (we already penalized above)
                    if curr_state['my_loc'] not in curr_state.get('my_eggs', set()):
                        base += 8.0
                    # CORNER EGG BONUS - Corner eggs give 2x score!
                    if self._is_corner(curr_state['my_loc']):
                        base += 15.0  # Extra bonus for corner eggs
                        # Track that we've been to this corner
                        if '_recent_corners' not in curr_state:
                            curr_state['_recent_corners'] = set()
                        curr_state['_recent_corners'].add(curr_state['my_loc'])
                elif mt == MoveType.TURD:
                    base += 25.0  # MUCH higher turd value in rollout
                    # Add strategic value
                    if hasattr(self, '_calculate_turd_value'):
                        try:
                            turd_value = self._calculate_turd_value(curr_state['my_loc'], curr_state) / 1000.0
                            base += turd_value
                        except:
                            pass

                base -= risk * 6.0

                # slightly discourage stepping back to last two positions
                sim_last = curr_state.get('last_loc')
                sim_two = curr_state.get('two_back_loc')
                if sim_last and new_loc == sim_last:
                    base *= 0.2
                if sim_two and new_loc == sim_two:
                    base *= 0.35

                # ... rest of existing logic computing my_space_after, etc ...
                tmp_state = copy.deepcopy(curr_state)
                if mt == MoveType.PLAIN:
                    tmp_state['my_loc'] = new_loc
                elif mt == MoveType.EGG or mt == MoveType.TURD:
                    if mt == MoveType.EGG:
                        tmp_state['my_eggs'] = tmp_state['my_eggs'].copy()
                        tmp_state['my_eggs'].add(curr_state['my_loc'])
                        # Corner eggs give CORNER_REWARD (3 total: 1 base + 2 bonus)
                        if self._is_corner(curr_state['my_loc']):
                            tmp_state['my_score'] += 3  # Corner eggs give 3 points total
                        else:
                            tmp_state['my_score'] += 1  # Normal eggs give 1 point
                        tmp_state['_recent_eggs'] = tmp_state.get('_recent_eggs', set()).copy()
                        tmp_state['_recent_eggs'].add(curr_state['my_loc'])
                        # Track recent corners
                        if '_recent_corners' not in tmp_state:
                            tmp_state['_recent_corners'] = set()
                        if self._is_corner(curr_state['my_loc']):
                            tmp_state['_recent_corners'].add(curr_state['my_loc'])
                    else:
                        tmp_state['my_turds'] = tmp_state['my_turds'].copy()
                        tmp_state['my_turds'].add(curr_state['my_loc'])
                        tmp_state['my_turds_left'] = tmp_state.get('my_turds_left', curr_state['my_turds_left']) - 1
                    tmp_state['my_loc'] = new_loc

                my_space_after = self._flood_fill_count(tmp_state, True)
                base += 0.5 * (my_space_after - self._flood_fill_count(curr_state, True))

                # ensure positive weight
                weighted.append((m, max(0.001, base)))

            
            total = sum(w for _, w in weighted)
            if total <= 0:
                # This is a fallback!
                best = random.choice(moves)
            else: 
                r = random.uniform(0, total)
                s = 0
                
                best = None
                for m, w in weighted:
                    s += w
                    if s >= r:
                        best = m
                        break
                if best is None:
                    best = max(weighted, key=lambda x: x[1])[0]

            curr_state = self._apply_move_sim(curr_state, best, True)

        # We evaluate the final state now
        return self.evaluate(curr_state)
    
    def mcts_backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _is_sim_state_valid(self, state: Dict) -> bool:
        try:
            # Must have locations!
            my_loc = state['my_loc']
            opp_loc = state['opp_loc']
            if not (0 <= my_loc[0] < 8 and 0 <= my_loc[1] < 8): return False
            if not (0 <= opp_loc[0] < 8 and 0 <= opp_loc[1] < 8): return False

            # Can't occupy opponent's egg/turd or vice versa!
            if my_loc in (state.get('opp_eggs', set()) | state.get('opp_turds', set())):
                return False
            if opp_loc in (state.get('my_eggs', set()) | state.get('my_turds', set())):
                return False

            # Turd counts are non-negative!
            if state.get('my_turds_left', 0) < 0 or state.get('opp_turds_left', 0) < 0:
                return False

            # No overlapping positions!
            if my_loc == opp_loc:
                return False

        except Exception:
            return False
        return True

    def _safe_state(self, board):
        return {
            'my_loc': board.chicken_player.get_location(),
            'opp_loc': board.chicken_enemy.get_location(),
            'my_score': board.chicken_player.eggs,
            'opp_score': board.chicken_enemy.eggs,
            'board': None   # or cached
        }
    
    def _calculate_invalid_state(self, state: Dict, is_me: bool) -> Dict:
        s = copy.deepcopy(state)
        if is_me:
            s['my_score'] -= 8
        else:
            s['opp_score'] -= 8
        return s

    def copy_state(self, state):
        # defensive deep copy of state dict
        return copy.deepcopy(state)
    
    def _dir_toward_center(self, loc):
        """Get direction toward center of board."""
        x, y = loc
        cx, cy = 3.5, 3.5

        dx = cx - x
        dy = cy - y

        if abs(dx) >= abs(dy):
            return Direction.RIGHT if dx > 0 else Direction.LEFT  # Fixed: RIGHT/LEFT for x-axis
        else:
            return Direction.DOWN if dy > 0 else Direction.UP  # Fixed: DOWN/UP for y-axis
        
    def _count_nearby_own_eggs_turds(self, loc: Tuple[int, int], state: Dict, radius: int = 2) -> int:
        """Count how many of our own eggs/turds are within radius of this location."""
        count = 0
        my_eggs = state.get('my_eggs', set())
        my_turds = state.get('my_turds', set())
        
        for egg in my_eggs:
            if manhattan_distance(loc, egg) <= radius:
                count += 1
        
        for turd in my_turds:
            if manhattan_distance(loc, turd) <= radius:
                count += 1
        
        return count
    
    def _is_contiguous_wall(self, turd_loc: Tuple[int, int], state: Dict) -> bool:
        """
        Check if placing a turd here creates or extends a contiguous orthogonal wall.
        Returns True if turd is adjacent (orthogonal) to at least one existing turd.
        """
        my_turds = state.get('my_turds', set())
        # Check orthogonal neighbors (not diagonals)
        ortho_neighbors = [
            (turd_loc[0] - 1, turd_loc[1]),  # Up
            (turd_loc[0] + 1, turd_loc[1]),  # Down
            (turd_loc[0], turd_loc[1] - 1),  # Left
            (turd_loc[0], turd_loc[1] + 1),  # Right
        ]
        for neighbor in ortho_neighbors:
            if neighbor in my_turds:
                return True
        return False
    
    def _count_wall_length(self, turd_loc: Tuple[int, int], state: Dict, direction: str) -> int:
        """
        Count the contiguous wall length if we place turd at turd_loc.
        Direction: 'horizontal', 'vertical', or 'both'.
        """
        my_turds = state.get('my_turds', set())
        count = 1  # Count the turd itself
        
        if direction in ['horizontal', 'both']:
            # Count left
            c = turd_loc[1] - 1
            while c >= 0 and (turd_loc[0], c) in my_turds:
                count += 1
                c -= 1
            # Count right
            c = turd_loc[1] + 1
            while c < 8 and (turd_loc[0], c) in my_turds:
                count += 1
                c += 1
        
        if direction in ['vertical', 'both']:
            # Count up
            r = turd_loc[0] - 1
            while r >= 0 and (r, turd_loc[1]) in my_turds:
                count += 1
                r -= 1
            # Count down
            r = turd_loc[0] + 1
            while r < 8 and (r, turd_loc[1]) in my_turds:
                count += 1
                r += 1
        
        return count
    
    def _detect_opponent_wall(self, state: Dict) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Detect if opponent has built a "wall" of turds/eggs across the board.
        Returns (is_walled, best_breakthrough_point).
        """
        opp_eggs = state.get('opp_eggs', set())
        opp_turds = state.get('opp_turds', set())
        opp_obstacles = opp_eggs | opp_turds
        
        if len(opp_obstacles) < self.WALL_DETECTION_THRESHOLD:
            return False, None
        
        # Check rows for horizontal walls
        row_densities = defaultdict(int)
        for loc in opp_obstacles:
            row_densities[loc[0]] += 1
        
        # Check columns for vertical walls
        col_densities = defaultdict(int)
        for loc in opp_obstacles:
            col_densities[loc[1]] += 1
        
        # Find rows/cols with dense obstacles (indicates wall-building)
        dense_rows = [r for r, count in row_densities.items() if count >= self.WALL_DETECTION_THRESHOLD]
        dense_cols = [c for c, count in col_densities.items() if count >= self.WALL_DETECTION_THRESHOLD]
        
        breakthrough_point = None
        # Check rows for gaps
        if dense_rows:
            for row in dense_rows:
                gap_count = 0
                gap_locs = []
                for col in range(8):
                    if (row, col) not in opp_obstacles and (row, col) not in state.get('my_eggs', set()):
                        gap_count += 1
                        gap_locs.append((row, col))
                # Prefer exploitable gaps
                if 0 < gap_count <= self.WALL_BREAK_CORRIDOR_WIDTH:
                    best_gap = min(gap_locs, key=lambda g: manhattan_distance(g, state['my_loc']))
                    if breakthrough_point is None or manhattan_distance(best_gap, state['my_loc']) < manhattan_distance(breakthrough_point, state['my_loc']):
                        breakthrough_point = best_gap
        
        # Check columns for gaps
        if dense_cols:
            for col in dense_cols:
                gap_count = 0
                gap_locs = []
                for row in range(8):
                    if (row, col) not in opp_obstacles and (row, col) not in state.get('my_eggs', set()):
                        gap_count += 1
                        gap_locs.append((row, col))
                if 0 < gap_count <= self.WALL_BREAK_CORRIDOR_WIDTH:
                    best_gap = min(gap_locs, key=lambda g: manhattan_distance(g, state['my_loc']))
                    if breakthrough_point is None or manhattan_distance(best_gap, state['my_loc']) < manhattan_distance(breakthrough_point, state['my_loc']):
                        breakthrough_point = best_gap
        
        return (breakthrough_point is not None), breakthrough_point
    
    def _get_wall_break_turd_value(self, turd_loc: Tuple[int, int], state: Dict, breakthrough_point: Tuple[int, int]) -> float:
        """Calculate bonus value for a turd that helps break through opponent wall."""
        bonus = 0.0
        opp_eggs = state.get('opp_eggs', set())
        opp_turds = state.get('opp_turds', set())
        opp_obstacles = opp_eggs | opp_turds
        
        # CRITICAL: Turds placed adjacent to breakthrough point reinforce the gap
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        if breakthrough_point:
            for dx, dy in deltas:
                adj_to_gap = (breakthrough_point[0] + dx, breakthrough_point[1] + dy)
                if 0 <= adj_to_gap[0] < 8 and 0 <= adj_to_gap[1] < 8:
                    if turd_loc == adj_to_gap and turd_loc not in opp_obstacles:
                        bonus += 100000.0
            
            # BONUS: Turds that widen the corridor through the wall
            dist_to_gap = manhattan_distance(turd_loc, breakthrough_point)
            if dist_to_gap <= 1:
                bonus += 50000.0
        
        # BONUS: Turds on the wall line (between us and them)
        if manhattan_distance(turd_loc, state['my_loc']) < manhattan_distance(turd_loc, state['opp_loc']):
            center = (3.5, 3.5)
            dist_to_center = manhattan_distance(turd_loc, center)
            if dist_to_center <= 3:
                wall_density = len([obs for obs in opp_obstacles if manhattan_distance(obs, turd_loc) <= 2])
                if wall_density >= 2:
                    bonus += 30000.0 * wall_density
        
        return bonus
    
    def _detect_territorial_circling(self, current_loc: Tuple[int, int], state: Dict) -> bool:
        """
        Detect if we're circling around our own territory with eggs/turds.
        Also detects turd-induced oscillation (bouncing due to blocked paths).
        Returns True if we're stuck in a tight loop.
        """
        # Only penalize territorial circling if there's actually free space to explore
        free_space = self._flood_fill_count(state, True)
        if free_space < 10:  # If board is mostly full, circling is okay
            return False
        
        # NEW: Detect turd-induced bouncing - when we're restricted to very few moves
        if len(self.position_history) >= 6:
            recent = self.position_history[-6:]
            unique_positions = len(set(recent))
            
            # If only visiting 2-3 positions repeatedly (turd walls blocking movement)
            if unique_positions <= 3:
                # Check if we're bouncing between the same positions
                if recent[-1] in recent[-4:-1]:
                    # We're revisiting a position from 2-4 moves ago - classic oscillation
                    return True
        
        # Check if we're making a "patrol pattern" (tight loop without progress)
        if len(self.position_history) >= 4:
            recent = self.position_history[-4:]
            if len(set(recent)) <= 2:
                return True
        
        return False
    
    def _calculate_turd_value(self, turd_loc: Tuple[int, int], state: Dict) -> float:
        """
        Calculate strategic value of placing a turd at this location.
        HEAVILY EMPHASIZES: solid orthogonal walls, blocking paths, trapping enemy.
        PENALIZES: weak diagonals, isolated turds, ineffective placements.
        """
        score = 0.0
        
        # FIRST: Check if opponent has built a wall and we can break it
        is_walled, breakthrough_point = self._detect_opponent_wall(state)
        if is_walled and breakthrough_point:
            wall_break_bonus = self._get_wall_break_turd_value(turd_loc, state, breakthrough_point)
            score += wall_break_bonus
            # If wall-breaking is viable, reduce other factors to focus on breaking through
            if wall_break_bonus > 20000:
                return score
        
        # CONTINUE with normal evaluation
        opp_loc = state['opp_loc']
        opp_eggs = state['opp_eggs']
        opp_turds = state['opp_turds']
        my_turds = state['my_turds']
        my_eggs = state['my_eggs']
        map_size = state.get('map_size', 8)
        
        # === WALL BUILDING STRATEGY (NEW AND IMPROVED) ===
        # STRONG BONUS: Reward orthogonal walls (solid blocking patterns)
        is_ortho_connected = self._is_contiguous_wall(turd_loc, state)
        if is_ortho_connected:
            # This turd extends an existing wall!
            # Bonus scales with wall length
            hor_length = self._count_wall_length(turd_loc, state, 'horizontal')
            ver_length = self._count_wall_length(turd_loc, state, 'vertical')
            
            # Longer walls = bigger bonus (up to 64000 for full-length wall)
            wall_bonus = max(hor_length, ver_length) * 8000
            score += wall_bonus
            
            # EXTRA BONUS: Reward walls that span multiple squares
            if hor_length >= 3:
                score += 20000  # Horizontal wall segment bonus
            if ver_length >= 3:
                score += 20000  # Vertical wall segment bonus
        
        # STRONG PENALTY: Punish isolated turds (not part of a wall)
        if not is_ortho_connected and my_turds:
            # This turd is isolated - bad strategy
            score -= 25000
        
        # BONUS: First turd gets special bonus (starting a wall)
        if not my_turds:
            score += 10000  # Encourage first turd placement
        
        # === DIAGONAL PENALTY (STRONG) ===
        # Penalize diagonal-only placements heavily
        if self._is_on_diagonal(turd_loc, map_size) and not is_ortho_connected:
            score -= 40000  # Heavy penalty for weak diagonal placement
        elif self._is_on_diagonal(turd_loc, map_size) and is_ortho_connected:
            # If diagonal AND part of a wall, only small penalty
            score -= 5000  # Minor penalty for diagonal orientation
        
        # === CRITICAL: Is enemy at spawn? TRAP THEM! ===
        opp_spawn = state.get('opp_spawn')
        if opp_spawn and opp_loc == opp_spawn:
            # Enemy just respawned from trapdoor - trap them!
            if manhattan_distance(turd_loc, opp_spawn) <= 2:
                score += self.W_TURD_TRAP_ENEMY_SPAWN
                # Extra bonus if this blocks their only escape route
                if manhattan_distance(turd_loc, opp_spawn) == 1:
                    score += 20000
        
        # === Is enemy in a corner? Reduce their escapes! ===
        if self._is_corner(opp_loc):
            dist_to_enemy = manhattan_distance(turd_loc, opp_loc)
            if dist_to_enemy <= 2:
                score += 30000 * (3 - dist_to_enemy)  # Closer = better
        
        # === Block path to valuable corner eggs ===
        corners = [(0,0), (0,7), (7,0), (7,7)]
        for corner in corners:
            # Only care about opponent's parity corners they haven't egged
            if (corner[0] + corner[1]) % 2 == self.opp_parity:
                if corner not in opp_eggs:
                    # This is a valuable corner (worth 3 eggs!)
                    dist_turd_to_corner = manhattan_distance(turd_loc, corner)
                    dist_enemy_to_corner = manhattan_distance(opp_loc, corner)
                    
                    # If we're blocking the path (closer to corner than enemy)
                    if dist_turd_to_corner <= 1 and dist_enemy_to_corner <= 4:
                        score += self.W_TURD_BLOCK_WINNING_PATH
        
        # === Calculate direct mobility reduction ===
        # How many squares does this turd block for the enemy?
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # Squares adjacent to this turd location become blocked
        blocked_squares = []
        for dx, dy in deltas:
            blocked_sq = (turd_loc[0] + dx, turd_loc[1] + dy)
            if 0 <= blocked_sq[0] < map_size and 0 <= blocked_sq[1] < map_size:
                blocked_squares.append(blocked_sq)
        
        # Count how many of these the enemy could have used
        enemy_blocked = 0
        for sq in blocked_squares:
            # Enemy can potentially use this square if it's empty
            if sq not in opp_eggs and sq not in my_turds and sq not in opp_turds and sq not in my_eggs:
                # Check if enemy can reach it (within reasonable distance)
                if manhattan_distance(opp_loc, sq) <= 5:
                    enemy_blocked += 1
        
        score += enemy_blocked * 3000  # Bonus for each square we block
        
        # === Territory reduction (flood fill comparison) ===
        try:
            opp_territory_before = self._flood_fill_count(state, is_me_perspective=False)
            
            # Simulate state with turd placed
            temp_state = state.copy()
            temp_state['my_turds'] = my_turds.copy() | {turd_loc}
            temp_state['my_eggs'] = my_eggs.copy()
            temp_state['opp_eggs'] = opp_eggs.copy()
            temp_state['opp_turds'] = opp_turds.copy()
            
            opp_territory_after = self._flood_fill_count(temp_state, is_me_perspective=False)
            
            territory_reduction = max(0, opp_territory_before - opp_territory_after)
            score += territory_reduction * self.W_TURD_REDUCE_TERRITORY
            
            # BONUS: If this traps enemy completely (reduces territory to near-zero)
            if opp_territory_after < 6 and opp_territory_before >= 6:
                score += 50000  # MASSIVE bonus for complete trap
        except:
            pass  # If flood fill fails, skip this component
        
        # === PENALTIES for bad placement ===
        
        # Don't place turds on edges (less effective)
        if self._is_on_edge(turd_loc, map_size):
            score -= 5000
        
        # Don't place turds far from enemy (not strategic)
        dist_to_enemy = manhattan_distance(turd_loc, opp_loc)
        if dist_to_enemy > 5:
            score -= 8000 * (dist_to_enemy - 5)  # Further = worse
        
        # Don't place turds in OUR territory (defensive = bad)
        nearby_our_stuff = self._count_nearby_own_eggs_turds(turd_loc, state, radius=2)
        if nearby_our_stuff >= 3:
            score -= 15000  # We're wasting turds defending our own area
        
        # === PATH BLOCKING ANALYSIS ===
        # Bonus for turds that block opponent shortest paths to key locations
        opp_escape_squares = self._get_opponent_escape_paths(opp_loc, state)
        blocked_escapes = 0
        for esc in opp_escape_squares:
            if manhattan_distance(turd_loc, esc) <= 1:
                blocked_escapes += 1
        score += blocked_escapes * 5000  # Bonus for each escape route blocked
        
        # === Bonus for central placement (more impactful) ===
        center = (3.5, 3.5)
        dist_to_center = manhattan_distance(turd_loc, center)
        if dist_to_center <= 3:
            score += 3000 * (4 - dist_to_center)  # Slightly increased center bonus
        
        return score
    
    def _get_opponent_escape_paths(self, opp_loc: Tuple[int, int], state: Dict) -> Set[Tuple[int, int]]:
        """
        Get the key escape squares the opponent needs to stay mobile.
        Returns adjacent squares that aren't blocked.
        """
        escape_squares = set()
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Orthogonal neighbors
        
        my_turds = state.get('my_turds', set())
        opp_eggs = state.get('opp_eggs', set())
        opp_turds = state.get('opp_turds', set())
        my_eggs = state.get('my_eggs', set())
        
        for dx, dy in deltas:
            sq = (opp_loc[0] + dx, opp_loc[1] + dy)
            if 0 <= sq[0] < 8 and 0 <= sq[1] < 8:
                # If square is empty, it's an escape route
                if sq not in my_turds and sq not in opp_eggs and sq not in opp_turds and sq not in my_eggs:
                    escape_squares.add(sq)
        
        return escape_squares

    def _set_probability_zero_except(self, trap_loc):
        (r, c) = trap_loc
        parity = (r + c) % 2
        grid = self.even_probs if parity == 0 else self.odd_probs

        for i in range(self.map_size):
            for j in range(self.map_size):
                if (i, j) == trap_loc:
                    grid[i, j] = 1.0
                else:
                    grid[i, j] = 0.0
    
    def _emergency_escape_move(self, loc):
        # if stuck and only trap moves remain: pick ANY direction that increases manhattan distance from trap
        best_move = None
        best_dist = -1
        if not self.tracker.confirmed_trap_squares:
            return (Direction.UP, MoveType.PLAIN)
        for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            new_loc = loc_after_direction(loc, d)
            if new_loc in self.tracker.confirmed_trap_squares:
                continue
            dist = min(manhattan_distance(new_loc, trap) for trap in self.tracker.confirmed_trap_squares)
            if dist > best_dist:
                best_dist = dist
                best_move = (d, MoveType.PLAIN)
        if best_move:
            return best_move
        return (Direction.UP, MoveType.PLAIN)
    
    def _get_enemy_territory_score(self, loc: Tuple[int, int], state: Dict) -> float:
        """Score bonus for invading enemy territory."""
        opp_spawn = state.get('opp_spawn')
        my_spawn = state.get('my_spawn')
        
        if not opp_spawn or not my_spawn:
            return 0.0
        
        dist_to_enemy_spawn = manhattan_distance(loc, opp_spawn)
        dist_to_my_spawn = manhattan_distance(loc, my_spawn)
        
        # Bonus for being closer to enemy spawn than our own
        if dist_to_enemy_spawn < dist_to_my_spawn:
            # DEEP invasion - the closer to enemy spawn, the better
            invasion_depth = (8 - dist_to_enemy_spawn)
            return self.W_INVADE_ENEMY_TERRITORY * (invasion_depth / 8.0)
        
        return 0.0

    def _count_denied_enemy_egg_squares(self, turd_loc: Tuple[int, int], state: Dict) -> int:
        """Count how many enemy-parity egg squares this turd would deny."""
        denied = 0
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # Squares adjacent to turd become inaccessible
        for dx, dy in deltas:
            adj = (turd_loc[0] + dx, turd_loc[1] + dy)
            if not (0 <= adj[0] < 8 and 0 <= adj[1] < 8):
                continue
            
            # If it's enemy parity and not already occupied
            if (adj[0] + adj[1]) % 2 == self.opp_parity:
                if adj not in state['opp_eggs'] and adj not in state['my_eggs']:
                    if adj not in state['my_turds'] and adj not in state['opp_turds']:
                        denied += 1
        
        return denied
    
    def _get_corner_denial_value(self, loc: Tuple[int, int], state: Dict) -> float:
        """Massive bonus for denying enemy access to their valuable corners."""
        score = 0.0
        corners = [(0,0), (0,7), (7,0), (7,7)]
        
        for corner in corners:
            # Only care about enemy's corners
            if (corner[0] + corner[1]) % 2 == self.opp_parity:
                if corner not in state['opp_eggs']:
                    # Enemy hasn't egged this corner yet
                    dist_to_corner = manhattan_distance(loc, corner)
                    
                    # If we're blocking the path (adjacent to corner)
                    if dist_to_corner == 1:
                        # Check if enemy can still reach it
                        enemy_dist = manhattan_distance(state['opp_loc'], corner)
                        if enemy_dist > dist_to_corner:
                            # We're closer - blocking access
                            score += self.W_CORNER_DENIAL
                    elif dist_to_corner == 2:
                        # Near the corner - good positioning
                        score += self.W_CORNER_DENIAL * 0.5
        
        return score
    
