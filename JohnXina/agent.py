from collections.abc import Callable
from time import time
from typing import List, Tuple, Dict, Set
import numpy as np
import math

from game.board import Board, manhattan_distance
from game.enums import Direction, MoveType, loc_after_direction
from game.game_map import prob_hear, prob_feel

class TimeoutError(Exception):
    """Custom exception raised when search budget is exceeded."""
    pass

class TrapdoorTracker:
    def __init__(self, map_size: int = 8):
        self.map_size = map_size
        self.even_probs = self._initialize_prior(0)
        self.odd_probs = self._initialize_prior(1)

    def _initialize_prior(self, parity: int) -> np.ndarray:
        dim = self.map_size
        prior = np.zeros((dim, dim))
        prior[2 : dim - 2, 2 : dim - 2] = 1.0
        prior[3 : dim - 3, 3 : dim - 3] = 2.0
        MIN_PRIOR_WEIGHT = 0.01 
        mask = np.indices((dim, dim)).sum(axis=0) % 2 == parity
        prior[mask & (prior == 0)] = MIN_PRIOR_WEIGHT 
        prior *= mask
        total = np.sum(prior)
        return prior / total if total > 0 else prior

    def update(self, current_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        readings = [(sensor_data[0], self.even_probs), (sensor_data[1], self.odd_probs)]
        x, y = current_loc
        for (heard, felt), belief_grid in readings:
            likelihood_grid = np.zeros_like(belief_grid)
            for r in range(self.map_size):
                for c in range(self.map_size):
                    if belief_grid[r, c] < 1e-10: continue 
                    dx, dy = abs(r - x), abs(c - y)
                    p_hear_val = prob_hear(dx, dy)
                    p_feel_val = prob_feel(dx, dy)
                    l_hear = p_hear_val if heard else (1.0 - p_hear_val)
                    l_feel = p_feel_val if felt else (1.0 - p_feel_val)
                    likelihood_grid[r, c] = l_hear * l_feel
            belief_grid *= likelihood_grid
            total = np.sum(belief_grid)
            if total > 0:
                belief_grid /= total

    def get_risk(self, loc: Tuple[int, int]) -> float:
        r, c = loc
        if (r + c) % 2 == 0:
            return self.even_probs[r, c]
        else:
            return self.odd_probs[r, c]


class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.tracker = TrapdoorTracker()
        
        # Heuristic Weights - TUNED
        self.W_EGG_DIFF = 5000.0       
        self.W_TERRITORY = 350.0       # Increased importance
        self.W_RISK = 8000.0           # Increased risk aversion
        self.W_BLOCK_WIN = 50000.0    
        self.W_DIAGONAL_CONTROL = 300.0 
        self.W_RETURN_PENALTY = 1500.0  # Stronger immediate reversal penalty
        self.W_OSCILLATION_PENALTY = 60000.0  # Stronger oscillation penalty
        self.W_CENTRAL_TURD = 650.0    
        self.CORNER_EGG_BONUS = 3.0

        self.W_TURD_COUNT_DIFF_BASE = 250.0  # Increased turd value
        self.W_EDGE_TURD_PENALTY = 150.0
        self.W_ADJACENT_TURD_PENALTY = 600.0
        self.W_CORNER_PROXIMITY = 200.0  # NEW: Reward being near corners

        # Territory Control Thresholds
        self.TERRITORY_MIN_THRESHOLD = 18  # Increased from 16
        self.W_TERRITORY_PENALTY = 3000.0  # Increased penalty
        
        # Trapdoor Safety
        self.TRAPDOOR_RISK_THRESHOLD = 0.008  # More conservative
        
        # Oscillation Detection - IMPROVED
        self.position_history = []  # Track last N positions
        self.HISTORY_LENGTH = 6     # Check for longer patterns
        
        # State Tracking
        self.last_loc = None 
        self.two_back_loc = None 
        
        # Search controls
        self.start_time = 0
        self.time_limit = 0
        self.my_parity = 0
        self.opp_parity = 0
        self.N_TOTAL_TURNS = 40
        
        # Cache for territory calculations
        self.territory_cache = {}
        self.cache_generation = 0

    def _is_on_diagonal(self, loc: Tuple[int, int], map_size: int = 8) -> bool:
        x, y = loc
        return x == y or x + y == map_size - 1

    def _is_on_edge(self, loc: Tuple[int, int], map_size: int = 8) -> bool:
        x, y = loc
        return x == 0 or x == map_size - 1 or y == 0 or y == map_size - 1

    def _is_corner(self, loc: Tuple[int, int], map_size: int = 8) -> bool:
        x, y = loc
        return (x == 0 or x == map_size - 1) and (y == 0 or y == map_size - 1)

    def _detect_oscillation(self, new_loc: Tuple[int, int]) -> bool:
        """Improved oscillation detection checking for repetitive patterns."""
        if len(self.position_history) < 4:
            return False
        
        # Check for immediate back-and-forth (A-B-A pattern)
        if len(self.position_history) >= 2:
            if new_loc == self.position_history[-2]:
                return True
        
        # Check for 3-cycle (A-B-C-A pattern)
        if len(self.position_history) >= 3:
            if new_loc == self.position_history[-3]:
                return True
        
        # Check if we're visiting the same small set of positions repeatedly
        if len(self.position_history) >= 6:
            recent = self.position_history[-6:]
            unique_positions = len(set(recent))
            if unique_positions <= 3:  # Only visiting 3 or fewer positions in last 6 moves
                return True
        
        return False

    def play(self, board_obj: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        self.start_time = time()
        
        self.my_parity = board_obj.chicken_player.even_chicken
        self.opp_parity = board_obj.chicken_enemy.even_chicken

        # Update Beliefs
        self.tracker.update(board_obj.chicken_player.get_location(), sensor_data)
        
        # Extract State
        state = self._extract_state(board_obj)
        
        # IMPROVED: Adaptive Time Budget
        total_time_rem = time_left()
        turns_rem = board_obj.turns_left_player
        
        # Phase-based time allocation
        if turns_rem > 30:  # Early game: explore more
            base_budget = 4.0
        elif turns_rem > 15:  # Mid game: balanced
            base_budget = 6.0
        else:  # End game: think harder
            base_budget = min(8.0, total_time_rem / max(1, turns_rem) * 0.85)
        
        # Ensure we don't run out of time
        safety_buffer = 1.0
        calculated_budget = min(base_budget, total_time_rem - safety_buffer)
        self.turn_budget = max(1.0, calculated_budget)
        self.time_limit = self.start_time + self.turn_budget

        # Clear territory cache periodically
        self.cache_generation += 1
        if self.cache_generation > 5:
            self.territory_cache.clear()
            self.cache_generation = 0

        # Iterative Deepening Minimax
        best_move = None
        valid_moves = board_obj.get_valid_moves()
        
        if not valid_moves:
            self.position_history.append(state['my_loc'])
            if len(self.position_history) > self.HISTORY_LENGTH:
                self.position_history.pop(0)
            return (Direction.UP, MoveType.PLAIN)

        # IMPROVED: Better move ordering
        valid_moves = self._order_moves(valid_moves, state)
        best_move = valid_moves[0]
        
        depth = 1
        MAX_DEPTH_CAP = 100
        best_depth = 0

        try:
            while True:
                # Leave 10% buffer
                if time() - self.start_time > self.turn_budget * 0.9:
                    break
                
                if depth > MAX_DEPTH_CAP:
                    break
                
                val, move = self.minimax(state, depth, float('-inf'), float('inf'), True)
                if move and board_obj.is_valid_move(move[0], move[1], False):
                    best_move = move
                    best_depth = depth

                depth += 1
                
        except TimeoutError:
            pass
        
        current_loc = state['my_loc']
        new_loc = loc_after_direction(current_loc, best_move[0]) if best_move else current_loc
        
        # IMPROVED: Better oscillation detection
        is_oscillating = self._detect_oscillation(new_loc)
        has_time_for_recompute = time_left() > 1.5
        
        if is_oscillating and has_time_for_recompute and len(valid_moves) > 1:
            # Try the second-best move instead of expensive re-search
            for alt_move in valid_moves[1:]:
                alt_loc = loc_after_direction(current_loc, alt_move[0])
                if not self._detect_oscillation(alt_loc):
                    best_move = alt_move
                    new_loc = alt_loc
                    break
        
        # Update position history
        self.position_history.append(new_loc)
        if len(self.position_history) > self.HISTORY_LENGTH:
            self.position_history.pop(0)
        
        self.two_back_loc = self.last_loc
        self.last_loc = new_loc

        print(f"TIME: {time() - self.start_time:.3f}s, Move: {best_move}, Depth: {best_depth}")
        return best_move

    def _order_moves(self, moves: List[Tuple[Direction, MoveType]], state: Dict) -> List[Tuple[Direction, MoveType]]:
        """Improved move ordering for better alpha-beta pruning."""
        def move_priority(move):
            d, m_type = move
            score = 0
            new_loc = loc_after_direction(state['my_loc'], d)
            
            # Prioritize egg moves
            if m_type == MoveType.EGG:
                score += 100
                # Extra bonus for corner eggs
                if self._is_corner(state['my_loc']):
                    score += 50
            
            # Strategic turd placement
            elif m_type == MoveType.TURD:
                score += 60
                # Prefer central/diagonal turds
                if self._is_on_diagonal(state['my_loc']):
                    score += 20
            
            # Avoid high-risk moves
            risk = self.tracker.get_risk(new_loc)
            score -= risk * 500
            
            # Prefer moves toward corners if we can lay eggs there
            if (state['my_loc'][0] + state['my_loc'][1]) % 2 == self.my_parity:
                corners = [(0,0), (0,7), (7,0), (7,7)]
                min_corner_dist = min(manhattan_distance(new_loc, c) for c in corners)
                score -= min_corner_dist * 5
            
            # Avoid positions that look like oscillation
            if new_loc in self.position_history[-4:]:
                score -= 200
            
            return score
        
        return sorted(moves, key=move_priority, reverse=True)

    def minimax(self, state, depth, alpha, beta, maximizing):
        if time() > self.time_limit:
            raise TimeoutError()
            
        if depth == 0:
            return self.evaluate(state), None

        moves = self._get_valid_moves_sim(state, is_me=maximizing)

        if not moves:
            return self.evaluate_terminal(state, maximizing), None

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
            
            risk = self.tracker.get_risk(next_loc)
            risks[move] = risk
            
            if risk <= self.TRAPDOOR_RISK_THRESHOLD:
                safe_moves.append(move)

        if safe_moves:
            moves_to_search = safe_moves
            best_move = safe_moves[0]
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
                score = 5
            return score
        return sorted(moves, key=priority, reverse=True)

    def evaluate(self, state):
        score = (state['my_score'] - state['opp_score']) * self.W_EGG_DIFF
        
        # Risk assessment
        risk = self.tracker.get_risk(state['my_loc'])
        score -= risk * self.W_RISK
        
        # Territory control with caching
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
        
        if my_space < self.TERRITORY_MIN_THRESHOLD:
            score -= self.W_TERRITORY_PENALTY
        
        # Diagonal control
        turd_control_score = 0
        map_size = state['map_size']
        for loc in state['my_turds']:
            if self._is_on_diagonal(loc, map_size):
                turd_control_score += 1
        for loc in state['opp_turds']:
            if self._is_on_diagonal(loc, map_size):
                turd_control_score -= 1
        score += turd_control_score * self.W_DIAGONAL_CONTROL
        
        # Central turd bonus and edge penalty
        central_turd_reward = 0
        edge_turd_penalty = 0
        for loc in state['my_turds']:
            x, y = loc
            if 2 <= x <= 5 and 2 <= y <= 5:
                central_turd_reward += 1
            elif self._is_on_edge(loc, map_size):
                edge_turd_penalty -= self.W_EDGE_TURD_PENALTY
        score += central_turd_reward * self.W_CENTRAL_TURD
        score += edge_turd_penalty
        
        # Oscillation penalties
        current_loc = state['my_loc']
        sim_last_loc = state.get('last_loc')
        sim_two_back_loc = state.get('two_back_loc')
        
        if sim_last_loc is not None and current_loc == sim_last_loc:
            score -= self.W_RETURN_PENALTY * 2
            
        if sim_two_back_loc is not None and current_loc == sim_two_back_loc:
            score -= self.W_OSCILLATION_PENALTY
        
        # Turd count differential (more valuable in endgame)
        turns_rem = state['turns_left_player']
        turd_value_multiplier = 1.0 + (1.0 - (turns_rem / self.N_TOTAL_TURNS))
        turd_diff_weight = self.W_TURD_COUNT_DIFF_BASE * turd_value_multiplier
        turd_diff = state['my_turds_left'] - state['opp_turds_left']
        score += turd_diff * turd_diff_weight
        
        # Corner egg bonus
        corners = {(0, 0), (0, 7), (7, 0), (7, 7)}
        my_corner_eggs = sum(1 for loc in state['my_eggs'] if loc in corners)
        opp_corner_eggs = sum(1 for loc in state['opp_eggs'] if loc in corners)
        score += (my_corner_eggs - opp_corner_eggs) * self.CORNER_EGG_BONUS * self.W_EGG_DIFF
        
        # NEW: Reward proximity to corners when we can lay eggs
        if (state['my_loc'][0] + state['my_loc'][1]) % 2 == self.my_parity:
            min_corner_dist = min(manhattan_distance(state['my_loc'], c) for c in corners)
            score -= min_corner_dist * self.W_CORNER_PROXIMITY / max(1, turns_rem / 10)
        
        # Adjacent turd penalty
        adjacent_penalty = 0
        my_turds_list = list(state['my_turds'])
        for i in range(len(my_turds_list)):
            for j in range(i + 1, len(my_turds_list)):
                if manhattan_distance(my_turds_list[i], my_turds_list[j]) == 1:
                    adjacent_penalty += self.W_ADJACENT_TURD_PENALTY
        score -= adjacent_penalty
        
        return score

    def evaluate_terminal(self, state, am_i_blocked):
        base = self.evaluate(state)
        return base - self.W_BLOCK_WIN if am_i_blocked else base + self.W_BLOCK_WIN

    def _flood_fill_count(self, state: Dict, is_me_perspective: bool) -> int:
        start_node = state['my_loc'] if is_me_perspective else state['opp_loc']
        
        if is_me_perspective:
            obstacles = state['opp_eggs']
            enemy_turds = state['opp_turds']
        else:
            obstacles = state['my_eggs']
            enemy_turds = state['my_turds']
        
        queue = [start_node]
        visited = {start_node}
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        while queue:
            curr = queue.pop(0)
            
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
                
        return len(visited)

    def _extract_state(self, board_obj: Board) -> Dict:
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
        }

    def _apply_move_sim(self, state: Dict, move: Tuple[Direction, MoveType], is_me: bool) -> Dict:
        ns = state.copy()
        ns['my_eggs'] = state['my_eggs'].copy()
        ns['opp_eggs'] = state['opp_eggs'].copy()
        ns['my_turds'] = state['my_turds'].copy()
        ns['opp_turds'] = state['opp_turds'].copy()

        d, m_type = move
        curr_loc = ns['my_loc'] if is_me else ns['opp_loc']
        
        if m_type == MoveType.EGG:
            if is_me:
                ns['my_eggs'].add(curr_loc)
                ns['my_score'] += 1
            else:
                ns['opp_eggs'].add(curr_loc)
                ns['opp_score'] += 1
        elif m_type == MoveType.TURD:
            if is_me:
                ns['my_turds'].add(curr_loc)
                ns['my_turds_left'] -= 1
            else:
                ns['opp_turds'].add(curr_loc)
                ns['opp_turds_left'] -= 1
                
        new_loc = loc_after_direction(curr_loc, d)
        
        if is_me:
            ns['my_loc'] = new_loc
            ns['two_back_loc'] = ns['last_loc']
            ns['last_loc'] = new_loc
        else:
            ns['opp_loc'] = new_loc
        
        return ns

    def _get_valid_moves_sim(self, state: Dict, is_me: bool) -> List[Tuple[Direction, MoveType]]:
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
        dist_to_enemy = manhattan_distance(curr_loc, other_loc)
        can_place_at_dist = dist_to_enemy > 1
        is_curr_loc_empty = (curr_loc not in my_eggs) and (curr_loc not in my_turds) and (curr_loc not in opp_eggs) and (curr_loc not in opp_turds)
        can_egg_step = ((cell_parity == required_parity) and is_curr_loc_empty and can_place_at_dist)
        
        has_adjacent_turd = False
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for dx, dy in deltas:
            adj_loc = (curr_loc[0] + dx, curr_loc[1] + dy)
            if adj_loc in my_turds:
                has_adjacent_turd = True
                break
        
        can_turd_step = ((turds_left > 0) and is_curr_loc_empty and can_place_at_dist and not has_adjacent_turd)
        
        for d in dirs:
            new_loc = loc_after_direction(curr_loc, d)
            nx, ny = new_loc
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue
            if new_loc == other_loc:
                continue
            if new_loc in opp_eggs or new_loc in opp_turds:
                continue
            if new_loc in my_eggs or new_loc in my_turds:
                continue
            
            in_turd_zone = False
            for tdx, tdy in deltas:
                if (nx + tdx, ny + tdy) in opp_turds:
                    in_turd_zone = True
                    break
            if in_turd_zone:
                continue
                
            moves.append((d, MoveType.PLAIN))
            if can_egg_step:
                moves.append((d, MoveType.EGG))
            if can_turd_step:
                moves.append((d, MoveType.TURD))
        return moves
