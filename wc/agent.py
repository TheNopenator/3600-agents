from collections.abc import Callable
from time import time
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import math
from collections import defaultdict

from game.board import Board, manhattan_distance
from game.enums import Direction, MoveType, loc_after_direction
from game.game_map import prob_hear, prob_feel

class TimeoutError(Exception):
    """Custom exception raised when search budget is exceeded."""
    pass

class TranspositionTable:
    """Stores previously evaluated positions to avoid recomputation."""
    def __init__(self, max_size=100000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_state(self, state: Dict) -> int:
        """Create a hash of the game state."""
        return hash((
            state['my_loc'],
            state['opp_loc'],
            frozenset(state['my_eggs']),
            frozenset(state['opp_eggs']),
            frozenset(state['my_turds']),
            frozenset(state['opp_turds']),
            state['my_turds_left'],
            state['opp_turds_left'],
        ))
    
    def get(self, state: Dict, depth: int, alpha: float, beta: float):
        """Retrieve cached evaluation if valid."""
        key = (self._hash_state(state), depth)
        if key in self.table:
            self.hits += 1
            stored_value, stored_flag, stored_move = self.table[key]
            if stored_flag == 'EXACT':
                return stored_value, stored_move
            elif stored_flag == 'LOWERBOUND' and stored_value >= beta:
                return stored_value, stored_move
            elif stored_flag == 'UPPERBOUND' and stored_value <= alpha:
                return stored_value, stored_move
        self.misses += 1
        return None, None
    
    def store(self, state: Dict, depth: int, value: float, flag: str, move):
        """Store evaluation in cache."""
        if len(self.table) >= self.max_size:
            keys_to_remove = list(self.table.keys())[:len(self.table)//4]
            for k in keys_to_remove:
                del self.table[k]
        
        key = (self._hash_state(state), depth)
        self.table[key] = (value, flag, move)
    
    def clear(self):
        self.table.clear()
        self.hits = 0
        self.misses = 0

class OpeningBook:
    """Stores optimal opening moves to save computation."""
    def __init__(self):
        self.book = self._build_book()
    
    def _build_book(self) -> Dict:
        """Build opening book with strong first moves."""
        book = {}
        
        # Left edge spawns (x=0, y=1-6)
        for y in range(1, 7):
            spawn = (0, y)
            if (0 + y) % 2 == 0:
                book[(spawn, 0)] = [(Direction.RIGHT, MoveType.EGG)]
            else:
                book[(spawn, 0)] = [(Direction.RIGHT, MoveType.PLAIN)]
            
            if (0 + y) % 2 == 1:
                book[(spawn, 1)] = [(Direction.RIGHT, MoveType.EGG)]
            else:
                book[(spawn, 1)] = [(Direction.RIGHT, MoveType.PLAIN)]
        
        # Right edge spawns (x=7, y=1-6)
        for y in range(1, 7):
            spawn = (7, y)
            if (7 + y) % 2 == 0:
                book[(spawn, 0)] = [(Direction.LEFT, MoveType.EGG)]
            else:
                book[(spawn, 0)] = [(Direction.LEFT, MoveType.PLAIN)]
            
            if (7 + y) % 2 == 1:
                book[(spawn, 1)] = [(Direction.LEFT, MoveType.EGG)]
            else:
                book[(spawn, 1)] = [(Direction.LEFT, MoveType.PLAIN)]
        
        return book
    
    def get_opening_move(self, loc: Tuple[int, int], parity: int, turn: int) -> Optional[Tuple[Direction, MoveType]]:
        """Get opening book move if available."""
        if turn == 0:
            moves = self.book.get((loc, parity))
            if moves:
                return moves[0]
        return None

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

            if np.max(belief_grid) == 1.0 and np.sum(belief_grid) == 1.0:
                 continue # Skip update if already confirmed

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


    # 🌟 NEW: HARD CONFIRMATION METHOD
    def confirm_trapdoor(self, loc: Tuple[int, int]):
        """Sets the probability of the given location to 1.0 and all others of that parity to 0.0."""
        r, c = loc
        parity = (r + c) % 2
        
        if parity == 0:
            grid = self.even_probs
        else:
            grid = self.odd_probs
            
        # Reset grid and set confirmed location to 1.0
        grid.fill(0.0)
        grid[r, c] = 1.0

    def get_risk(self, loc: Tuple[int, int]) -> float:
        r, c = loc
        # ⚠️ NEW: Check for confirmed trapdoor first
        if loc in self.confirmed_trapdoors:
            return 1.0 # Absolute risk
            
        if r < 2 or r > 5 or c < 2 or c > 5:
            return 0.0
            
        if (r + c) % 2 == 0: return self.even_probs[r, c]
        else: return self.odd_probs[r, c]


class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.tracker = TrapdoorTracker()

        # 🌟 NEW: CONFIRMED TRAPDOOR SET
        self.tracker.confirmed_trapdoors = set() 
        self.transposition_table = TranspositionTable()
        self.opening_book = OpeningBook()
        
        self.confirmed_trapdoors = self.tracker.confirmed_trapdoors # Reference for convenience

        self.transposition_table = TranspositionTable()
        self.opening_book = OpeningBook()
        
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.visited_squares = set()
        
        # NEW: Strategic quadrant control
        self.chosen_quadrant = None  # 'top' or 'bottom'
        self.quadrant_filled = False
        
        # Heuristic Weights - RETUNED FOR WINNING STRATEGY
        self.W_EGG_DIFF = 5000.0       
        self.W_TERRITORY = 250.0
        self.W_RISK_BASE = 8000.0
        self.W_BLOCK_WIN = 50000.0    
        self.W_DIAGONAL_CONTROL = 300.0 
        self.W_RETURN_PENALTY = 1500.0  
        self.W_OSCILLATION_PENALTY = 60000.0
        self.W_CENTRAL_TURD = 650.0    
        self.CORNER_EGG_BONUS = 3.0

        self.W_TURD_COUNT_DIFF_BASE = 250.0  
        self.W_EDGE_TURD_PENALTY = 150.0
        self.W_ADJACENT_TURD_PENALTY = 600.0
        self.W_CORNER_PROXIMITY = 150.0
        self.W_MOBILITY = 100.0
        self.W_OPPONENT_MOBILITY = 150.0
        
        # Egg-focused weights
        self.W_ABSOLUTE_EGG_COUNT = 800.0
        self.W_EGG_OPPORTUNITY = 3000.0
        self.W_EXPLORATION = 120.0
        self.W_PATH_TO_EGG = 180.0
        self.W_TERRITORY_UTILIZATION = 400.0
        
        # NEW: Quadrant strategy weights
        self.W_QUADRANT_BONUS = 2000.0       # Bonus for being in target quadrant
        self.W_CENTER_RUSH = 1500.0          # Bonus for moving toward center
        self.W_EGG_WHILE_RUSHING = 3000.0    # Bonus for egg steps while moving to center
        self.W_CENTRAL_TURD_PLACEMENT = 5000.0  # HUGE bonus for turds in center area

        self.TERRITORY_MIN_THRESHOLD = 16
        self.W_TERRITORY_PENALTY = 2500.0
        self.TRAPDOOR_RISK_THRESHOLD_BASE = 0.008
        
        self.last_score = 0
        self.aspiration_window = 200
        self.QUIESCENCE_DEPTH = 6
        self.NULL_MOVE_REDUCTION = 2
        self.LMR_THRESHOLD = 4
        self.LMR_REDUCTION = 1
        
        self.position_history = []
        self.HISTORY_LENGTH = 6
        
        self.spawn_loc = board.chicken_player.get_spawn() 
        self.last_loc = self.spawn_loc # Initializing last_loc correctly
        self.two_back_loc = None 
        
        self.start_time = 0
        self.time_limit = 0
        self.my_parity = 0
        self.opp_parity = 0
        self.N_TOTAL_TURNS = 40
        
        self.territory_cache = {}
        self.cache_generation = 0
        self.nodes_searched = 0
        
        # Define center area for turd placement (2-5, 2-5)
        self.CENTER_AREA = {(x, y) for x in range(2, 6) for y in range(2, 6)}
        
        # Define quadrants
        self.TOP_QUADRANT = {(x, y) for x in range(8) for y in range(0, 4)}
        self.BOTTOM_QUADRANT = {(x, y) for x in range(8) for y in range(4, 8)}

    def _initialize_key_squares(self) -> Dict[str, Set[Tuple[int, int]]]:
        """Define strategically important squares for territory control."""
        return {
            'center': {(3, 3), (3, 4), (4, 3), (4, 4)},
            'inner_ring': {(2, 2), (2, 3), (2, 4), (2, 5), 
                          (3, 2), (3, 5), (4, 2), (4, 5),
                          (5, 2), (5, 3), (5, 4), (5, 5)},
            'main_diagonal': {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)},
            'anti_diagonal': {(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)},
            'vertical_divider': {(3, y) for y in range(8)} | {(4, y) for y in range(8)},
            'horizontal_divider': {(x, 3) for x in range(8)} | {(x, 4) for x in range(8)},
        }
    
    def _choose_quadrant(self, my_spawn: Tuple[int, int], opp_loc: Tuple[int, int]) -> str:
        """Choose which quadrant to prioritize (top or bottom)."""
        # If opponent is in top half, we go bottom, and vice versa
        # This maximizes our territory while minimizing conflict
        if opp_loc[1] < 4:  # Opponent in top half
            return 'bottom'
        else:
            return 'top'
    
    def _is_quadrant_filled(self, state: Dict, quadrant: str) -> bool:
        """Check if our chosen quadrant is mostly filled with eggs."""
        target_area = self.TOP_QUADRANT if quadrant == 'top' else self.BOTTOM_QUADRANT
        
        # Count available egg squares in our quadrant
        available_in_quadrant = 0
        for loc in target_area:
            if (loc[0] + loc[1]) % 2 != self.my_parity:
                continue
            if loc in state['my_eggs'] or loc in state['opp_eggs']:
                continue
            if loc in state['my_turds'] or loc in state['opp_turds']:
                continue
            available_in_quadrant += 1
        
        # If less than 5 squares available, quadrant is "filled"
        return available_in_quadrant < 5
    
    def _get_target_quadrant(self, state: Dict) -> str:
        """Get the current target quadrant based on game state."""
        # First time: choose quadrant
        if self.chosen_quadrant is None:
            self.chosen_quadrant = self._choose_quadrant(
                state.get('my_spawn', (0, 0)),
                state['opp_loc']
            )
            print(f"📍 Chose {self.chosen_quadrant.upper()} quadrant strategy")
        
        # Check if we need to switch to opposite quadrant
        if not self.quadrant_filled and self._is_quadrant_filled(state, self.chosen_quadrant):
            self.quadrant_filled = True
            self.chosen_quadrant = 'bottom' if self.chosen_quadrant == 'top' else 'top'
            print(f"🔄 Switching to {self.chosen_quadrant.upper()} quadrant")
        
        return self.chosen_quadrant

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
        
        if len(self.position_history) >= 2:
            if new_loc == self.position_history[-2]:
                return True
        
        if len(self.position_history) >= 3:
            if new_loc == self.position_history[-3]:
                return True
        
        if len(self.position_history) >= 6:
            recent = self.position_history[-6:]
            unique_positions = len(set(recent))
            if unique_positions <= 3:
                return True
        
        return False
    
    def _get_game_phase(self, turns_taken: int, turns_left: int) -> str:
        """Determine current game phase."""
        if turns_taken < 12:
            return 'EXPANSION'
        elif turns_taken < 25:
            return 'CONSOLIDATION'
        else:
            return 'OPTIMIZATION'

    def play(self, board_obj: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        self.start_time = time()
        self.nodes_searched = 0
        
        self.my_parity = board_obj.chicken_player.even_chicken
        self.opp_parity = board_obj.chicken_enemy.even_chicken

        # 🌟 BUG FIX: TRAPDOOR CONFIRMATION LOGIC
        current_pos = board_obj.chicken_player.get_location()
        is_at_spawn = current_pos == self.spawn_loc

        # Check if the current location is the spawn, and the last location was in the center.
        if self.last_loc and self.last_loc != current_pos and is_at_spawn and self.last_loc in self.CENTER_AREA:
            print(f"🚨 TRAPDOOR CONFIRMED at {self.last_loc}! Updating tracker.")
            self.confirmed_trapdoors.add(self.last_loc)
            self.tracker.confirm_trapdoor(self.last_loc) 
            # Note: We don't update self.last_loc yet, it will be updated later with the new move

        self.tracker.update(current_pos, sensor_data)
        self.visited_squares.add(current_pos)
        
        turn_num = self.N_TOTAL_TURNS - board_obj.turns_left_player
        opening_move = self.opening_book.get_opening_move(
            board_obj.chicken_player.get_location(),
            self.my_parity,
            turn_num
        )
        
        if opening_move and board_obj.is_valid_move(opening_move[0], opening_move[1], False):
            print(f"Using opening book move: {opening_move}")
            self.position_history.append(board_obj.chicken_player.get_location())
            if len(self.position_history) > self.HISTORY_LENGTH:
                self.position_history.pop(0)
            return opening_move
        
        state = self._extract_state(board_obj)
        
        # Detect game phase and quadrant
        turns_taken = self.N_TOTAL_TURNS - board_obj.turns_left_player
        phase = self._get_game_phase(turns_taken, board_obj.turns_left_player)
        target_quadrant = self._get_target_quadrant(state)
        
        # ENDGAME DETECTION
        turns_left = board_obj.turns_left_player
        available_eggs = self._count_available_egg_squares(state, is_me=True)
        is_endgame = turns_left < 15 or available_eggs < 10
        is_desperate = turns_left < 8 or available_eggs < 5
        
        # Check if we're oscillating
        is_oscillating = self._detect_oscillation(current_pos)
        
        # ENDGAME ANTI-TRAP: If oscillating in endgame, force exploration
        if is_endgame and is_oscillating:
            print(f"⚠️ ENDGAME TRAP DETECTED! Turns: {turns_left}, Eggs available: {available_eggs}")
            exploration_move = self._find_exploration_move(board_obj, state)
            if exploration_move:
                print(f"🚀 Using exploration move: {exploration_move}")
                self.position_history.append(current_pos)
                if len(self.position_history) > self.HISTORY_LENGTH:
                    self.position_history.pop(0)
                return exploration_move
        
        # # DESPERATION MODE: Find any path to an egg square
        # if is_desperate and available_eggs > 0:
        #     desperate_move = self._find_path_to_any_egg_square(board_obj, state)
        #     if desperate_move and board_obj.is_valid_move(desperate_move[0], desperate_move[1], False):
        #         print(f"💀 DESPERATION MODE: Forcing path to egg square")
        #         self.position_history.append(current_pos)
        #         if len(self.position_history) > self.HISTORY_LENGTH:
        #             self.position_history.pop(0)
        #         return desperate_move
        
        total_time_rem = time_left()
        turns_rem = board_obj.turns_left_player
        
        if turns_rem > 30:  
            base_budget = 10
        elif turns_rem > 20:  
            base_budget = 10
        elif turns_rem > 10:  
            base_budget = 10
        elif turns_rem > 5:
            base_budget = min(10.0, total_time_rem / max(1, turns_rem) * 0.90)
        else:
            base_budget = min(15.0, total_time_rem / max(1, turns_rem) * 0.95)
        
        safety_buffer = 0.8
        calculated_budget = min(base_budget, total_time_rem - safety_buffer)
        self.turn_budget = max(1.0, calculated_budget)
        self.time_limit = self.start_time + self.turn_budget

        self.cache_generation += 1
        if self.cache_generation > 5:
            self.territory_cache.clear()
            self.cache_generation = 0

        valid_moves = board_obj.get_valid_moves()
        
        if not valid_moves:
            self.position_history.append(state['my_loc'])
            if len(self.position_history) > self.HISTORY_LENGTH:
                self.position_history.pop(0)
            return (Direction.UP, MoveType.PLAIN)

        # 🥚 CRITICAL FIX: Always prefer EGG moves when on egg square
        # If we're on an egg square and can lay eggs, ONLY consider egg moves
        current_loc = board_obj.chicken_player.get_location()
        can_lay_egg_here = board_obj.can_lay_egg()
        
        if can_lay_egg_here:
            # Filter to only egg moves
            egg_moves = [move for move in valid_moves if move[1] == MoveType.EGG]
            
            if egg_moves:
                print(f"🥚 ON EGG SQUARE - Only considering EGG moves ({len(egg_moves)} available)")
                valid_moves = egg_moves
            else:
                # Shouldn't happen, but just in case
                print(f"⚠️ Can lay egg but no egg moves available? Using all moves.")
        
        valid_moves = self._order_moves(valid_moves, state)
        best_move = valid_moves[0]
        
        depth = 1
        MAX_DEPTH_CAP = 500
        best_depth = 0

        try:
            while True:
                if time() - self.start_time > self.turn_budget * 0.92:
                    break
                
                if depth > MAX_DEPTH_CAP:
                    break
                
                if depth > 3 and self.last_score != 0:
                    alpha = self.last_score - self.aspiration_window
                    beta = self.last_score + self.aspiration_window
                    val, move = self.minimax(state, depth, alpha, beta, True)
                    
                    if val <= alpha or val >= beta:
                        val, move = self.minimax(state, depth, float('-inf'), float('inf'), True)
                else:
                    val, move = self.minimax(state, depth, float('-inf'), float('inf'), True)
                
                if move and board_obj.is_valid_move(move[0], move[1], False):
                    best_move = move
                    best_depth = depth
                    self.last_score = val
                    self.history_table[move] += depth * depth

                depth += 1
                
        except TimeoutError:
            pass
        
        # 🛡️ SAFETY CHECK: If we ended up with a PLAIN move but EGG is legal, force EGG
        if best_move and best_move[1] == MoveType.PLAIN and can_lay_egg_here:
            direction = best_move[0]
            # Check if (direction, EGG) is in valid moves
            egg_alternative = (direction, MoveType.EGG)
            if egg_alternative in valid_moves:
                print(f"⚠️ CORRECTING: Forcing EGG move instead of PLAIN in direction {direction}")
                best_move = egg_alternative
        
        current_loc = state['my_loc']
        new_loc = loc_after_direction(current_loc, best_move[0]) if best_move else current_loc
        
        is_oscillating = self._detect_oscillation(new_loc)
        has_time_for_recompute = time_left() > 1.5
        
        if best_move and best_move[1] == MoveType.EGG:
            is_oscillating = False
        
        if is_oscillating and has_time_for_recompute and len(valid_moves) > 1:
            for alt_move in valid_moves[1:]:
                alt_loc = loc_after_direction(current_loc, alt_move[0])
                if not self._detect_oscillation(alt_loc) or alt_move[1] == MoveType.EGG:
                    best_move = alt_move
                    new_loc = alt_loc
                    break
        
        # History Update
        current_loc_before_move = state['my_loc'] # The location we started this turn at
        new_loc = loc_after_direction(current_loc_before_move, best_move[0]) if best_move else current_loc_before_move
        
        self.position_history.append(new_loc)
        if len(self.position_history) > self.HISTORY_LENGTH: self.position_history.pop(0)
        # 🌟 Store the planned *next* location as last_loc for the *next* turn's trap check
        self.last_loc = new_loc 

        # 🌟 REQUESTED DEBUG OUTPUT: TRAPDOOR PROBABILITIES
        self._print_neighbor_probabilities(current_pos, best_move)

        return best_move
    

    # 🌟 NEW: DEBUG PRINT METHOD
    def _print_neighbor_probabilities(self, current_loc: Tuple[int, int], best_move: Tuple[Direction, MoveType]):
        """Prints the trapdoor risk for all neighbors and the planned next move."""
        print("-" * 40)
        print(f"Current Location: {current_loc}")
        print(f"Selected Move: {best_move[0].name} {best_move[1].name}")
        
        locations_to_check = {current_loc: "CURRENT"}
        
        # Add neighbors
        dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        for d in dirs:
            loc = loc_after_direction(current_loc, d)
            if 0 <= loc[0] < 8 and 0 <= loc[1] < 8:
                locations_to_check[loc] = d.name
                
        # Add the location of the next move (if not already a neighbor)
        next_loc = loc_after_direction(current_loc, best_move[0])
        if next_loc not in locations_to_check:
            locations_to_check[next_loc] = f"{best_move[0].name} (Next)"
            
        print("Trapdoor Risk for Neighbors:")
        for loc, desc in locations_to_check.items():
            risk = self.tracker.get_risk(loc)
            # Find the actual probability in the grid, as get_risk returns 0.0 for safe zones
            r, c = loc
            prob_in_grid = 0.0
            if r >= 2 and r <= 5 and c >= 2 and c <= 5:
                if (r + c) % 2 == 0: prob_in_grid = self.tracker.even_probs[r, c]
                else: prob_in_grid = self.tracker.odd_probs[r, c]
            
            # Highlight confirmed trapdoors or high risk
            highlight = ""
            if loc in self.confirmed_trapdoors:
                 highlight = " ***CONFIRMED TRAP***"
            elif risk > 0.15:
                 highlight = " **HIGH RISK**"
            elif prob_in_grid < 0.001 and loc not in self.confirmed_trapdoors:
                 highlight = " (STATISTICALLY CLEAR)"

            print(f"  {desc.ljust(15)} {loc}: Risk={risk:.4f} (Prob: {prob_in_grid:.4f}){highlight}")
        print("-" * 40)
    

    def _find_exploration_move(self, board_obj: Board, state: Dict) -> Optional[Tuple[Direction, MoveType]]:
        """Find a move that breaks oscillation by going to unvisited/less-visited areas."""
        valid_moves = board_obj.get_valid_moves()
        if not valid_moves:
            return None
        
        current_loc = board_obj.chicken_player.get_location()
        
        # Score moves by exploration value
        exploration_scores = []
        for move in valid_moves:
            direction, move_type = move
            next_loc = loc_after_direction(current_loc, direction)
            
            score = 0
            
            # Heavily prefer unvisited squares
            if next_loc not in self.visited_squares:
                score += 1000
            
            # Prefer squares far from recent positions
            if len(self.position_history) > 0:
                min_dist_to_history = min(
                    manhattan_distance(next_loc, hist_loc) 
                    for hist_loc in self.position_history[-6:]
                )
                score += min_dist_to_history * 200
            
            # Prefer moves that lead toward egg squares
            if (next_loc[0] + next_loc[1]) % 2 == self.my_parity:
                score += 500
                # Even better if it's an egg move
                if move_type == MoveType.EGG:
                    score += 2000
            
            # Prefer moves away from center (explore edges/corners)
            dist_from_center = manhattan_distance(next_loc, (3, 3)) + manhattan_distance(next_loc, (4, 4))
            score += dist_from_center * 50
            
            # Avoid recently visited squares
            if next_loc in self.position_history[-3:]:
                score -= 500
            
            # Lower risk tolerance in exploration (we're desperate)
            risk = self.tracker.get_risk(next_loc)
            score -= risk * 300  # Reduced from 500
            
            exploration_scores.append((move, score))
        
        # Return the most exploratory move
        exploration_scores.sort(key=lambda x: x[1], reverse=True)
        return exploration_scores[0][0] if exploration_scores else None
    
    def _find_path_to_any_egg_square(self, board_obj: Board, state: Dict) -> Optional[Tuple[Direction, MoveType]]:
        """Use BFS to find shortest path to ANY available egg square."""
        current_loc = board_obj.chicken_player.get_location()
        
        # Find all available egg squares
        available_egg_squares = []
        for x in range(8):
            for y in range(8):
                loc = (x, y)
                if (x + y) % 2 != self.my_parity:
                    continue
                if loc in state['my_eggs'] or loc in state['opp_eggs']:
                    continue
                if loc in state['my_turds'] or loc in state['opp_turds']:
                    continue
                available_egg_squares.append(loc)
        
        if not available_egg_squares:
            return None
        
        # BFS to find path to nearest egg square
        queue = [(current_loc, [])]  # (location, path of directions)
        visited = {current_loc}
        
        while queue:
            loc, path = queue.pop(0)
            
            # Check if we reached an egg square
            if loc in available_egg_squares:
                # We found a path! Return the first move
                if len(path) > 0:
                    first_direction = path[0]
                    # Check if we can lay an egg at current location
                    if board_obj.can_lay_egg():
                        return (first_direction, MoveType.EGG)
                    else:
                        return (first_direction, MoveType.PLAIN)
                else:
                    # We're already on an egg square!
                    if board_obj.can_lay_egg():
                        # Just move and lay
                        valid_moves = board_obj.get_valid_moves()
                        egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
                        if egg_moves:
                            return egg_moves[0]
            
            # Explore neighbors
            for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
                next_loc = loc_after_direction(loc, direction)
                
                # Check if valid and not visited
                if not board_obj.is_valid_cell(next_loc):
                    continue
                if next_loc in visited:
                    continue
                
                # Check if blocked (use simplified check)
                if next_loc == state['opp_loc']:
                    continue
                if next_loc in state['opp_eggs'] or next_loc in state['opp_turds']:
                    continue
                if next_loc in state['my_eggs'] or next_loc in state['my_turds']:
                    continue
                
                # Check turd zones
                blocked_by_turd = False
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    adj = (next_loc[0] + dx, next_loc[1] + dy)
                    if adj in state['opp_turds']:
                        blocked_by_turd = True
                        break
                
                if blocked_by_turd:
                    continue
                
                visited.add(next_loc)
                queue.append((next_loc, path + [direction]))
        
        # No path found, return None
        return None

    def _order_moves(self, moves: List[Tuple[Direction, MoveType]], state: Dict) -> List[Tuple[Direction, MoveType]]:
        """Move ordering following winning strategy: egg steps toward center, turds in middle."""
        turns_taken = self.N_TOTAL_TURNS - state['turns_left_player']
        target_quadrant = self._get_target_quadrant(state)
        
        # Check if we're on an egg square
        current_loc = state['my_loc']
        on_egg_square = (current_loc[0] + current_loc[1]) % 2 == self.my_parity
        
        def move_priority(move):
            score = 0
            new_loc = loc_after_direction(current_loc, move[0])
            direction, move_type = move
            
            if move in self.killer_moves.get(0, []):
                score += 10000
            
            score += self.history_table.get(move, 0)
            
            # 🥚 ABSOLUTE PRIORITY: If on egg square, EGG moves get MASSIVE bonus
            if on_egg_square and move_type == MoveType.EGG:
                score += 100000  # This should ALWAYS win
            elif on_egg_square and move_type == MoveType.PLAIN:
                score -= 50000  # Heavy penalty for plain moves on egg squares
            
            # WINNING STRATEGY IMPLEMENTATION
            
            # 1. EGG STEPS while rushing to center/quadrant (HIGHEST PRIORITY early game)
            if move_type == MoveType.EGG and turns_taken < 20:
                score += 10000  # Very high priority
                
                # Extra bonus if moving toward center
                current_center_dist = manhattan_distance(current_loc, (3, 3)) + manhattan_distance(current_loc, (4, 4))
                new_center_dist = manhattan_distance(new_loc, (3, 3)) + manhattan_distance(new_loc, (4, 4))
                if new_center_dist < current_center_dist:
                    score += 2000  # Moving toward center
                
                # Extra bonus if in target quadrant
                target_area = self.TOP_QUADRANT if target_quadrant == 'top' else self.BOTTOM_QUADRANT
                if current_loc in target_area:
                    score += 1500
            
            elif move_type == MoveType.EGG:
                score += 5000  # Still good in late game
                if self._is_corner(current_loc):
                    score += 1000
            
            # 2. TURDS: Only in center area (2-5, 2-5), NEVER on edges
            if move_type == MoveType.TURD:
                if current_loc in self.CENTER_AREA:
                    score += 8000  # VERY high priority for central turds
                    
                    # Extra bonus for true center (3-4, 3-4)
                    if current_loc in {(3, 3), (3, 4), (4, 3), (4, 4)}:
                        score += 2000
                    
                elif self._is_on_edge(current_loc):
                    score -= 5000  # STRONGLY avoid edge turds
                    
                else:
                    score += 2000  # Okay in other areas
            
            # 3. PLAIN MOVES: Toward center and target quadrant
            if move_type == MoveType.PLAIN:
                # Moving toward center
                current_center_dist = manhattan_distance(current_loc, (3, 3)) + manhattan_distance(current_loc, (4, 4))
                new_center_dist = manhattan_distance(new_loc, (3, 3)) + manhattan_distance(new_loc, (4, 4))
                if new_center_dist < current_center_dist:
                    score += 1000
                
                # Moving into target quadrant
                target_area = self.TOP_QUADRANT if target_quadrant == 'top' else self.BOTTOM_QUADRANT
                if new_loc in target_area and current_loc not in target_area:
                    score += 1500
                
                # Moving toward egg square
                if (new_loc[0] + new_loc[1]) % 2 == self.my_parity:
                    score += 500
            
            # Risk assessment
            risk = self.tracker.get_risk(new_loc)
            score -= risk * 300  # Less risk-averse in this aggressive strategy
            
            # Avoid oscillation
            if new_loc in self.position_history[-4:] and move_type != MoveType.EGG:
                score -= 200
            
            # Mobility
            potential_moves = self._count_potential_moves(new_loc, state, is_me=True)
            score += potential_moves * 10
            
            return score
        
        return sorted(moves, key=move_priority, reverse=True)
    
    def _find_nearest_egg_square(self, loc: Tuple[int, int], state: Dict) -> Optional[Tuple[int, int]]:
        """Find nearest unoccupied square where we can lay an egg."""
        best_dist = float('inf')
        best_square = None
        
        for x in range(8):
            for y in range(8):
                square = (x, y)
                if (x + y) % 2 != self.my_parity:
                    continue
                if square in state['my_eggs'] or square in state['opp_eggs']:
                    continue
                if square in state['my_turds'] or square in state['opp_turds']:
                    continue
                
                dist = manhattan_distance(loc, square)
                if dist < best_dist:
                    best_dist = dist
                    best_square = square
        
        return best_square
    
    def _count_potential_moves(self, loc: Tuple[int, int], state: Dict, is_me: bool) -> int:
        """Count how many moves would be available from a position."""
        count = 0
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        
        other_loc = state['opp_loc'] if is_me else state['my_loc']
        opp_eggs = state['opp_eggs'] if is_me else state['my_eggs']
        opp_turds = state['opp_turds'] if is_me else state['my_turds']
        my_eggs = state['my_eggs'] if is_me else state['opp_eggs']
        my_turds = state['my_turds'] if is_me else state['opp_turds']
        
        for d in directions:
            new_loc = loc_after_direction(loc, d)
            nx, ny = new_loc
            
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue
            if new_loc == other_loc:
                continue
            if new_loc in opp_eggs or new_loc in opp_turds:
                continue
            if new_loc in my_eggs or new_loc in my_turds:
                continue
            
            deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            in_turd_zone = False
            for tdx, tdy in deltas:
                if (nx + tdx, ny + tdy) in opp_turds:
                    in_turd_zone = True
                    break
            
            if not in_turd_zone:
                count += 1
        
        return count
    
    def _order_moves_sim(self, moves: List[Tuple[Direction, MoveType]], state: Dict, is_me: bool) -> List[Tuple[Direction, MoveType]]:
        """Fast move ordering for simulation with killer/history heuristics."""
        def priority(move):
            score = 0
            
            for depth_key in range(0, 5):
                if move in self.killer_moves.get(depth_key, []):
                    score += 5000 - depth_key * 100
                    break
            
            score += self.history_table.get(move, 0) / 10
            
            if move[1] == MoveType.EGG:
                score += 100
            elif move[1] == MoveType.TURD:
                score += 50
            
            current_loc = state['my_loc'] if is_me else state['opp_loc']
            next_loc = loc_after_direction(current_loc, move[0])
            potential_moves = self._count_potential_moves(next_loc, state, is_me)
            score += potential_moves * 5
            
            return score
        
        return sorted(moves, key=priority, reverse=True)

    def minimax(self, state, depth, alpha, beta, maximizing):
        self.nodes_searched += 1
        
        if time() > self.time_limit:
            raise TimeoutError()
        
        tt_value, tt_move = self.transposition_table.get(state, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move
            
        if depth == 0:
            return self.quiescence_search(state, self.QUIESCENCE_DEPTH, alpha, beta, maximizing), None

        moves = self._get_valid_moves_sim(state, is_me=maximizing)

        if not moves:
            terminal_eval = self.evaluate_terminal(state, maximizing)
            self.transposition_table.store(state, depth, terminal_eval, 'EXACT', None)
            return terminal_eval, None

        if maximizing:
            moves = self._order_moves_sim(moves, state, True)
        else:
            moves = self._order_moves_sim(moves, state, False)
        
        best_move = moves[0]
        safe_moves = []
        risks = {}
        
        for move in moves:
            current_loc = state['my_loc'] if maximizing else state['opp_loc']
            next_loc = loc_after_direction(current_loc, move[0])
            
            risk = self.tracker.get_risk(next_loc)
            risks[move] = risk
            
            if risk <= self.TRAPDOOR_RISK_THRESHOLD_BASE:
                safe_moves.append(move)

        if safe_moves:
            moves_to_search = safe_moves
            best_move = safe_moves[0]
        else:
            moves_to_search = sorted(moves, key=lambda m: risks[m])[:12]
            best_move = moves_to_search[0]

        if maximizing:
            max_eval = float('-inf')
            
            if depth >= 3 and not self._is_in_check(state, True):
                null_state = state.copy()
                null_eval, _ = self.minimax(null_state, depth - 1 - self.NULL_MOVE_REDUCTION, alpha, beta, False)
                if null_eval >= beta:
                    return beta, None
            
            move_count = 0
            for move in moves_to_search:
                new_state = self._apply_move_sim(state, move, is_me=True)
                
                reduction = 0
                if move_count >= self.LMR_THRESHOLD and depth >= 3 and move[1] == MoveType.PLAIN:
                    reduction = self.LMR_REDUCTION
                
                eval_val, _ = self.minimax(new_state, depth - 1 - reduction, alpha, beta, False)
                
                if reduction > 0 and eval_val > alpha:
                    eval_val, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop()
                    break
                
                move_count += 1
            
            flag = 'EXACT' if max_eval > alpha and max_eval < beta else ('LOWERBOUND' if max_eval >= beta else 'UPPERBOUND')
            self.transposition_table.store(state, depth, max_eval, flag, best_move)
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            
            move_count = 0
            for move in moves_to_search:
                new_state = self._apply_move_sim(state, move, is_me=False)
                
                reduction = 0
                if move_count >= self.LMR_THRESHOLD and depth >= 3 and move[1] == MoveType.PLAIN:
                    reduction = self.LMR_REDUCTION
                
                eval_val, _ = self.minimax(new_state, depth - 1 - reduction, alpha, beta, True)
                
                if reduction > 0 and eval_val < beta:
                    eval_val, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                
                beta = min(beta, eval_val)
                if beta <= alpha:
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop()
                    break
                
                move_count += 1
            
            flag = 'EXACT' if min_eval > alpha and min_eval < beta else ('LOWERBOUND' if min_eval >= beta else 'UPPERBOUND')
            self.transposition_table.store(state, depth, min_eval, flag, best_move)
            
            return min_eval, best_move
    
    def quiescence_search(self, state: Dict, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Search only tactical moves (eggs/turds) to avoid horizon effect."""
        stand_pat = self.evaluate(state)
        
        if depth == 0:
            return stand_pat
        
        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            
            moves = self._get_valid_moves_sim(state, is_me=True)
            tactical_moves = [m for m in moves if m[1] in (MoveType.EGG, MoveType.TURD)]
            
            for move in tactical_moves:
                new_state = self._apply_move_sim(state, move, is_me=True)
                score = self.quiescence_search(new_state, depth - 1, alpha, beta, False)
                
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            
            return alpha
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
            
            moves = self._get_valid_moves_sim(state, is_me=False)
            tactical_moves = [m for m in moves if m[1] in (MoveType.EGG, MoveType.TURD)]
            
            for move in tactical_moves:
                new_state = self._apply_move_sim(state, move, is_me=False)
                score = self.quiescence_search(new_state, depth - 1, alpha, beta, True)
                
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            
            return beta
    
    def _is_in_check(self, state: Dict, is_me: bool) -> bool:
        """Check if player is in a constrained position (for null move pruning)."""
        moves = self._get_valid_moves_sim(state, is_me=is_me)
        return len(moves) < 3

    def evaluate(self, state):
        turns_taken = self.N_TOTAL_TURNS - state['turns_left_player']
        game_phase = turns_taken / self.N_TOTAL_TURNS
        
        score = (state['my_score'] - state['opp_score']) * self.W_EGG_DIFF
        
        absolute_egg_weight = self.W_ABSOLUTE_EGG_COUNT * (1.0 - game_phase * 0.6)
        score += state['my_score'] * absolute_egg_weight
        
        available_egg_squares = self._count_available_egg_squares(state, is_me=True)
        opp_available = self._count_available_egg_squares(state, is_me=False)
        
        # ENDGAME URGENCY: Massively penalize having few egg opportunities
        if available_egg_squares < 5:
            penalty_multiplier = 2.0 if game_phase > 0.7 else 1.0
            score -= (5 - available_egg_squares) * self.W_EGG_OPPORTUNITY * penalty_multiplier
        
        # ENDGAME: Being on an egg square is critical
        current_loc = state['my_loc']
        if game_phase > 0.6 and (current_loc[0] + current_loc[1]) % 2 == self.my_parity:
            score += 3000  # Big bonus for being on an egg square in endgame
        
        score += (available_egg_squares - opp_available) * 150
        
        risk_multiplier = 1.0 - (game_phase * 0.6)
        
        egg_deficit = state['opp_score'] - state['my_score']
        if egg_deficit > 3:
            risk_multiplier *= 0.6
        elif egg_deficit > 0:
            risk_multiplier *= 0.8
        
        # ENDGAME: Be even more risk-tolerant when desperate
        if game_phase > 0.75 and available_egg_squares < 8:
            risk_multiplier *= 0.5  # Take big risks to find eggs
        
        risk = self.tracker.get_risk(state['my_loc'])
        dynamic_risk_weight = self.W_RISK_BASE * risk_multiplier
        score -= risk * dynamic_risk_weight
        
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
        
        if my_space > 0:
            utilization = len(state['my_eggs']) / max(1, my_space)
            score += utilization * self.W_TERRITORY_UTILIZATION
        
        if my_space < self.TERRITORY_MIN_THRESHOLD:
            score -= self.W_TERRITORY_PENALTY
        
        spawn_dist = manhattan_distance(state['my_loc'], state.get('my_spawn', (0, 0)))
        score += spawn_dist * self.W_EXPLORATION
        
        # ENDGAME: Path to egg square becomes CRITICAL
        if (state['my_loc'][0] + state['my_loc'][1]) % 2 != self.my_parity:
            nearest_egg = self._find_nearest_egg_square(state['my_loc'], state)
            if nearest_egg:
                dist_to_egg = manhattan_distance(state['my_loc'], nearest_egg)
                # In endgame, being far from egg squares is very bad
                endgame_multiplier = 1.0 if game_phase < 0.6 else 2.5
                score -= dist_to_egg * self.W_PATH_TO_EGG * (1 + game_phase) * endgame_multiplier
        
        my_mobility = self._count_potential_moves(state['my_loc'], state, is_me=True)
        opp_mobility = self._count_potential_moves(state['opp_loc'], state, is_me=False)
        score += (my_mobility - opp_mobility) * self.W_MOBILITY
        score -= opp_mobility * self.W_OPPONENT_MOBILITY
        
        turd_control_score = 0
        map_size = state['map_size']
        for loc in state['my_turds']:
            if self._is_on_diagonal(loc, map_size):
                turd_control_score += 1
        for loc in state['opp_turds']:
            if self._is_on_diagonal(loc, map_size):
                turd_control_score -= 1
        score += turd_control_score * self.W_DIAGONAL_CONTROL
        
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
        
        blocking_bonus = 0
        opp_parity = 1 - self.my_parity
        for loc in state['my_turds']:
            deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            for dx, dy in deltas:
                adj = (loc[0] + dx, loc[1] + dy)
                if 0 <= adj[0] < 8 and 0 <= adj[1] < 8:
                    if (adj[0] + adj[1]) % 2 == opp_parity:
                        blocking_bonus += 1
        score += blocking_bonus * 80
        
        sim_last_loc = state.get('last_loc')
        sim_two_back_loc = state.get('two_back_loc')
        
        is_egg_square = (current_loc[0] + current_loc[1]) % 2 == self.my_parity
        oscillation_multiplier = 0.3 if is_egg_square else 1.0
        
        # ENDGAME: Oscillation penalties are less important than finding eggs
        if game_phase > 0.7:
            oscillation_multiplier *= 0.5
        
        if sim_last_loc is not None and current_loc == sim_last_loc:
            score -= self.W_RETURN_PENALTY * 2 * oscillation_multiplier
            
        if sim_two_back_loc is not None and current_loc == sim_two_back_loc:
            score -= self.W_OSCILLATION_PENALTY * oscillation_multiplier
        
        turns_rem = state['turns_left_player']
        turd_value_multiplier = 0.5 + (game_phase * 1.5)
        turd_diff_weight = self.W_TURD_COUNT_DIFF_BASE * turd_value_multiplier
        turd_diff = state['my_turds_left'] - state['opp_turds_left']
        score += turd_diff * turd_diff_weight
        
        corners = {(0, 0), (0, 7), (7, 0), (7, 7)}
        my_corner_eggs = sum(1 for loc in state['my_eggs'] if loc in corners)
        opp_corner_eggs = sum(1 for loc in state['opp_eggs'] if loc in corners)
        score += (my_corner_eggs - opp_corner_eggs) * self.CORNER_EGG_BONUS * self.W_EGG_DIFF
        
        adjacent_penalty = 0
        my_turds_list = list(state['my_turds'])
        for i in range(len(my_turds_list)):
            for j in range(i + 1, len(my_turds_list)):
                if manhattan_distance(my_turds_list[i], my_turds_list[j]) == 1:
                    adjacent_penalty += self.W_ADJACENT_TURD_PENALTY
        score -= adjacent_penalty
        
        egg_metric = 0
        if game_phase < 0.5:
            unique_rows = len(set(egg[1] for egg in state['my_eggs']))
            unique_cols = len(set(egg[0] for egg in state['my_eggs']))
            egg_metric = (unique_rows + unique_cols) * 40
        else:
            for egg in state['my_eggs']:
                neighbors = 0
                deltas = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dx, dy in deltas:
                    adj = (egg[0] + dx, egg[1] + dy)
                    if adj in state['my_eggs']:
                        neighbors += 1
                egg_metric += neighbors * 15
        score += egg_metric
        
        return score
    
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
            'my_spawn': board_obj.chicken_player.get_spawn(),
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