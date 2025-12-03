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

class ImprovedTrapdoorTracker:
    """Enhanced trapdoor tracking with binary safe/danger classification."""
    def __init__(self, map_size: int = 8, decay: float = 0.98, smooth_sigma: float = 0.8):
        self.map_size = map_size
        self.decay = decay
        self.smooth_sigma = smooth_sigma

        self.even_prior = self._initialize_prior(0)
        self.odd_prior = self._initialize_prior(1)

        # Convert prior to log-odds!
        eps = 1e-12
        self.even_logodds = np.log(np.clip(self.even_prior, eps, 1 - eps) / np.clip(1 - self.even_prior, eps, 1 - eps))
        self.odd_logodds = np.log(np.clip(self.odd_prior, eps, 1 - eps) / np.clip(1 - self.odd_prior, eps, 1 - eps))
        
        # Binary classification
        self.confirmed_safe = set()  # We've been here, no trigger
        self.confirmed_danger = set()  # Multiple signals, likely trap
        self.potential_danger = set()  # Some signals, be cautious
        
        # Signal tracking
        self.visit_counts = defaultdict(int)
        self.no_signal_counts = defaultdict(int)
        
        # Thresholds!
        self.POTENTIAL_THRESHOLD = 0.20
        self.CONFIRMED_THRESHOLD = 0.75
        self.SAFE_VISIT_THRESHOLD = 3
        self.MAX_LOGODDS = 50.0

    def _initialize_prior(self, parity: int) -> np.ndarray:
        """Initialize probability distribution for trapdoors."""
        dim = self.map_size
        prior = np.zeros((dim, dim), dtype=float)
        
        # Higher weight in center (matches game's trapdoor placement)
        prior[2 : dim - 2, 2 : dim - 2] = 1.0
        prior[3 : dim - 3, 3 : dim - 3] = 2.0
        
        # Apply parity mask
        mask = (np.indices((dim, dim)).sum(axis=0) % 2) == parity
        prior *= mask.astype(float)
        
        # Normalize
        total = prior.sum()
        if total <= 0:
            prior[:] = 1.0
            total = prior.sum()
        
        prior = prior / total

        # Slightly inflate priors to avoid extremes!
        prior = 0.01 + 0.98 * prior
        prior = prior / prior.sum()
        return prior

    def update(self, current_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        """Update trapdoor beliefs based on sensor readings."""
        x, y = current_loc
        self.visit_counts[current_loc] += 1
    
        # Process sensor data
        heard_even, felt_even = sensor_data[0]
        heard_odd, felt_odd = sensor_data[1]
        
        # if we're physically on a square and didn't trigger, mark safe evidence
        # only mark fully confirmed safe after multiple safe visits
        if not (heard_even or felt_even or heard_odd or felt_odd):
            self.no_signal_counts[current_loc] += 1
            if self.no_signal_counts[current_loc] >= self.SAFE_VISIT_THRESHOLD:
                self.confirmed_safe.add(current_loc)
                if current_loc in self.confirmed_danger:
                    self.confirmed_danger.remove(current_loc)
                if current_loc in self.potential_danger:
                    self.potential_danger.discard(current_loc)
        else:
            # Positive local signals reduce safe counts!
            self.no_signal_counts[current_loc] = 0
        
        # Update Bayesian beliefs for unexplored areas
        self._update_bayesian(current_loc, sensor_data)

        self.even_logodds *= self.decay
        self.odd_logodds *= self.decay

        self._spatial_smooth()

        np.clip(self.even_logodds, -self.MAX_LOGODDS, self.MAX_LOGODDS, out=self.even_logodds)
        np.clip(self.odd_logodds, -self.MAX_LOGODDS, self.MAX_LOGODDS, out=self.odd_logodds)

        self._refresh_sets_from_posteriors()
    
    def _process_positive_signals(self, loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        """Process when we detect trapdoor signals."""
        x, y = loc
        
        heard_even, felt_even = sensor_data[0]
        heard_odd, felt_odd = sensor_data[1]
        
        # Map signal strength to danger zones
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                sq = (x + dx, y + dy)
                if not (0 <= sq[0] < 8 and 0 <= sq[1] < 8):
                    continue
                if sq in self.confirmed_safe:
                    continue
                
                # Check which parity and signal type
                sq_parity = (sq[0] + sq[1]) % 2
                
                if sq_parity == 0:  # Even square
                    if heard_even:
                        self.signal_history[sq]['hear'] += 1
                    if felt_even:
                        self.signal_history[sq]['feel'] += 2  # Feel is more reliable
                else:  # Odd square
                    if heard_odd:
                        self.signal_history[sq]['hear'] += 1
                    if felt_odd:
                        self.signal_history[sq]['feel'] += 2
                
                # Classify based on accumulated signals
                total_signals = self.signal_history[sq]['hear'] + self.signal_history[sq]['feel']
                
                if total_signals >= 6:
                    self.confirmed_danger.add(sq)
                    if sq in self.potential_danger:
                        self.potential_danger.remove(sq)
                elif total_signals >= 3:
                    if sq not in self.confirmed_danger:
                        self.potential_danger.add(sq)
    
    def _process_negative_signals(self, loc: Tuple[int, int]):
        """Process when we hear/feel nothing - mark nearby squares as safer."""
        x, y = loc
        
        # Adjacent and nearby squares are likely safe
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                sq = (x + dx, y + dy)
                if not (0 <= sq[0] < 8 and 0 <= sq[1] < 8):
                    continue
                if sq == loc:
                    continue
                
                # Reduce danger ratings for nearby squares
                if sq in self.potential_danger:
                    self.signal_history[sq]['hear'] = max(0, self.signal_history[sq]['hear'] - 1)
                    self.signal_history[sq]['feel'] = max(0, self.signal_history[sq]['feel'] - 1)
                    
                    total = self.signal_history[sq]['hear'] + self.signal_history[sq]['feel']
                    if total < 3:
                        self.potential_danger.remove(sq)
    
    def _update_bayesian(self, current_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        """Update Bayesian probability grid (used for unexplored squares)."""
        readings = [(sensor_data[0], self.even_prior), (sensor_data[1], self.odd_prior)]
        x, y = current_loc
        dim = self.map_size
        eps = 1e-12
        
        # Generate distance grid once!
        xs = np.arange(dim)[:, None]
        ys = np.arange(dim)[None, :]
        dx = np.abs(xs - x)
        dy = np.abs(ys - y)
        dists = np.maximum(dx, dy)

        # We'll vectorize using arrays of prob_hear and prob_feel values.
        p_hear = np.zeros((dim, dim), dtype=float)
        p_feel = np.zeros((dim, dim), dtype=float)
        for r in range(dim):
            for c in range(dim):
                p_hear[r, c] = prob_hear(abs(r - x), abs(c - y))
                p_feel[r, c] = prob_feel(abs(r - x), abs(c - y))

        # Unpack boolean observations!
        (heard_even, felt_even), (heard_odd, felt_odd) = sensor_data

        # For even parity, update!
        even_mask = ((xs + ys) % 2 == 0)
        odd_mask = ~even_mask
        
        le_hear = p_hear.copy()
        le_feel = p_feel.copy()
        P_obs_given_trap_even = np.where(heard_even, le_hear, 1.0 - le_hear) * np.where(felt_even, le_feel, 1.0 - le_feel)

        # For "no trap," assume sensors will occasionally generate false positives with small base rates!
        small_noise_hear = 0.02
        small_noise_feel = 0.01
        P_obs_given_notrap_even = np.where(heard_even, small_noise_hear, 1.0 - small_noise_hear) * np.where(felt_even, small_noise_feel, 1.0 - small_noise_feel)

        # Compute LR for even parity (avoid division by zero)!
        LR_even = (P_obs_given_trap_even + eps) / (P_obs_given_notrap_even + eps)
        # Similarly for odd parity using the odd observation!
        lo_hear = p_hear.copy()
        lo_feel = p_feel.copy()
        P_obs_given_trap_odd = np.where(heard_odd, lo_hear, 1.0 - lo_hear) * np.where(felt_odd, lo_feel, 1.0 - lo_feel)
        P_obs_given_notrap_odd = np.where(heard_odd, small_noise_hear, 1.0 - small_noise_hear) * np.where(felt_odd, small_noise_feel, 1.0 - small_noise_feel)
        LR_odd = (P_obs_given_trap_odd + eps) / (P_obs_given_notrap_odd + eps)

        # Convert LR to log-LR
        logLR_even = np.log(LR_even)
        logLR_odd  = np.log(LR_odd)

        # Add logLR to proper parity logodds arrays!
        # In practice we only add to even_logodds where mask is True, and odd where false!
        self.even_logodds += logLR_even * even_mask
        self.odd_logodds += logLR_odd  * odd_mask

        # Additional local 'feel' is more reliable: if felt is True at location x,y, strongly increase that cell
        # (this addresses immediate evidence on current square)
        if felt_even and ((x + y) % 2 == 0):
            self.even_logodds[x, y] += 2.0  # strong bump
        if felt_odd and ((x + y) % 2 == 1):
            self.odd_logodds[x, y] += 2.0

    def _spatial_smooth(self):
        # 3x3 kernal smoothing!
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1, 0.4, 0.1],
                           [0.05, 0.1, 0.05]])
        def smooth_grid(logodds):
            # Convert logodds -> prob, smooth, convert back!
            probs = 1.0 / (1.0 + np.exp(-logodds))
            newp = probs.copy()
            dim = self.map_size
            for r in range(dim):
                for c in range(dim):
                    acc = 0.0
                    wsum = 0.0
                    for kr in range(-1, 2):
                        for kc in range(-1, 2):
                            rr = r + kr
                            cc = c + kc
                            if 0 <= rr < dim and 0 <= cc < dim:
                                weight = kernel[kr + 1, kc + 1]
                                acc += probs[rr, cc] * weight
                                wsum += weight
                    if wsum > 0:
                        newp[r, c] = acc / wsum
            eps = 1e-12
            newp = np.clip(newp, eps, 1.0 - eps)
            return np.log(newp / (1.0 - newp))
        self.even_logodds = smooth_grid(self.even_logodds)
        self.odd_logodds = smooth_grid(self.odd_logodds)

    def _refresh_sets_from_posteriors(self):
        """Map posterior probabilities to confirmed/potential sets using thresholds."""
        # get posterior probs
        even_p = 1.0 / (1.0 + np.exp(-self.even_logodds))
        odd_p  = 1.0 / (1.0 + np.exp(-self.odd_logodds))

        self.confirmed_danger.clear()
        self.potential_danger.clear()
        # scan grid
        for r in range(self.map_size):
            for c in range(self.map_size):
                p = even_p[r, c] if ((r + c) % 2 == 0) else odd_p[r, c]
                loc = (r, c)
                # skip squares we've visited and observed safe repeatedly
                if self.visit_counts.get(loc, 0) >= self.SAFE_VISIT_THRESHOLD and self.no_signal_counts.get(loc, 0) >= self.SAFE_VISIT_THRESHOLD:
                    # treat as safe override
                    continue
                if p >= self.CONFIRMED_THRESHOLD:
                    self.confirmed_danger.add(loc)
                elif p >= self.POTENTIAL_THRESHOLD:
                    self.potential_danger.add(loc)
                # otherwise neither set contains it
    
    def get_posterior(self, loc: Tuple[int, int]) -> float:
        r, c = loc
        if not (0 <= r < self.map_size and 0 <= c < self.map_size):
            return 0.0
        if (r + c) % 2 == 0:
            p = 1.0 / (1.0 + math.exp(-float(self.even_logodds[r, c])))
        else:
            p = 1.0 / (1.0 + math.exp(-float(self.odd_logodds[r, c])))
        return float(p)

    def get_risk(self, loc: Tuple[int, int], game_phase: float = 0.5) -> float:
        """
        Get risk score for a location.
        Returns: 0.0 (safe) to 1.0 (very dangerous)
        
        game_phase: 0.0 (early) to 1.0 (late) - be more aggressive late game
        """
        # Confirmed safe = no risk
        if loc in self.confirmed_safe:
            return 0.0
        
        # Confirmed danger = high risk
        if loc in self.confirmed_danger:
            return 0.98
        
        # Potential danger = medium risk
        if loc in self.potential_danger:
            return 0.65
        
        p = self.get_posterior(loc)
        # Scale down posterior risk slightly in later phases to be more aggressive!
        risk_multiplier = 1.0 - (game_phase * 0.45)
        return float(min(p * risk_multiplier, 0.6))
    
    def is_safe_zone(self, loc: Tuple[int, int]) -> bool:
        """Check if a location is in a safe zone (confirmed or very likely)."""
        if loc in self.confirmed_safe:
            return True
        # Also safe if posterior tiny and many surrounding safes!
        p = self.get_posterior(loc)
        if p < 0.02:
            # Check if surrounded by safe squares
            safe_neighbors = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (loc[0] + dx, loc[1] + dy)
                    nx, ny = loc[0] + dx, loc[1] + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if (nx, ny) in self.confirmed_safe:
                            safe_neighbors += 1
            return safe_neighbors >= 4 # If 4+ neighbors are safe, likely safe
        return False


class TerritoryManager:
    """Manages territory control and strategic division of the board."""
    def __init__(self, my_parity: int):
        self.my_parity = my_parity
        self.opp_parity = 1 - my_parity
        
        # Territory division
        self.my_territory = set()
        self.contested_territory = set()
        self.opp_territory = set()
        
        # Wall construction - ENHANCED for larger territory
        self.wall_lines = []  # Multiple wall lines for better control
        self.wall_type = None  # 'vertical' or 'horizontal'
        self.wall_target_squares = set()  # Egg squares on the walls
        self.secondary_wall_squares = set()  # Secondary defensive line
        self.wall_completed = False
        
        # Strategic zones
        self.safe_quadrant = None  # 'top', 'bottom', 'left', 'right'
        self.claimed_percentage = 0.0
    
    def initialize_strategy(self, my_loc: Tuple[int, int], opp_loc: Tuple[int, int], turn: int):
        """
        Determine territory strategy based on initial positions.
        NOW AIMS TO CAPTURE 50%+ OF THE BOARD with multi-line defense.
        """
        # Determine natural division based on spawn positions
        x_dist = abs(my_loc[0] - opp_loc[0])
        y_dist = abs(my_loc[1] - opp_loc[1])
        
        # Choose division axis and BUILD MULTIPLE WALLS
        if x_dist > y_dist:
            # Vertical division - claim MORE territory
            self.wall_type = 'vertical'
            
            # Determine our side and be AGGRESSIVE
            if my_loc[0] < opp_loc[0]:
                self.safe_quadrant = 'left'
                # Claim 50% of board: columns 0-3 (50% = 4 columns)
                # PRIMARY wall at column 3 (our rightmost boundary)
                # SECONDARY wall at column 2 (backup defense)
                primary_wall = 3
                secondary_wall = 2
                
                self.wall_lines = [primary_wall, secondary_wall]
                self._mark_territory_vertical(0, 4, mine=True)  # Claim columns 0-3
                self._mark_territory_vertical(4, 8, mine=False)  # Leave columns 4-7 to opponent
                
            else:
                self.safe_quadrant = 'right'
                # Claim 50% of board: columns 4-7 (50% = 4 columns)
                # PRIMARY wall at column 4 (our leftmost boundary)
                # SECONDARY wall at column 5 (backup defense)
                primary_wall = 4
                secondary_wall = 5
                
                self.wall_lines = [primary_wall, secondary_wall]
                self._mark_territory_vertical(4, 8, mine=True)  # Claim columns 4-7
                self._mark_territory_vertical(0, 4, mine=False)  # Leave columns 0-3 to opponent
            
            # Build DOUBLE WALL - primary and secondary lines
            for y in range(8):
                # Primary wall (outer boundary)
                wall_sq = (primary_wall, y)
                if (primary_wall + y) % 2 == self.my_parity:
                    self.wall_target_squares.add(wall_sq)
                
                # Secondary wall (backup defense)
                wall_sq2 = (secondary_wall, y)
                if (secondary_wall + y) % 2 == self.my_parity:
                    self.secondary_wall_squares.add(wall_sq2)
        
        else:
            # Horizontal division - claim MORE territory
            self.wall_type = 'horizontal'
            
            if my_loc[1] < opp_loc[1]:
                self.safe_quadrant = 'top'
                # Claim 50% of board: rows 0-3
                primary_wall = 3
                secondary_wall = 2
                
                self.wall_lines = [primary_wall, secondary_wall]
                self._mark_territory_horizontal(0, 4, mine=True)  # Claim rows 0-3
                self._mark_territory_horizontal(4, 8, mine=False)  # Leave rows 4-7
                
            else:
                self.safe_quadrant = 'bottom'
                # Claim 50% of board: rows 4-7
                primary_wall = 4
                secondary_wall = 5
                
                self.wall_lines = [primary_wall, secondary_wall]
                self._mark_territory_horizontal(4, 8, mine=True)  # Claim rows 4-7
                self._mark_territory_horizontal(0, 4, mine=False)  # Leave rows 0-3
            
            # Build DOUBLE WALL
            for x in range(8):
                # Primary wall
                wall_sq = (x, primary_wall)
                if (x + primary_wall) % 2 == self.my_parity:
                    self.wall_target_squares.add(wall_sq)
                
                # Secondary wall
                wall_sq2 = (x, secondary_wall)
                if (x + secondary_wall) % 2 == self.my_parity:
                    self.secondary_wall_squares.add(wall_sq2)
        
        # Calculate claimed percentage
        total_squares = 64
        claimed = len(self.my_territory)
        self.claimed_percentage = claimed / total_squares
    
    def _mark_territory_vertical(self, x_start: int, x_end: int, mine: bool):
        """Mark vertical strip as territory."""
        for x in range(x_start, x_end):
            for y in range(8):
                if mine:
                    self.my_territory.add((x, y))
                else:
                    self.opp_territory.add((x, y))
    
    def _mark_territory_horizontal(self, y_start: int, y_end: int, mine: bool):
        """Mark horizontal strip as territory."""
        for y in range(y_start, y_end):
            for x in range(8):
                if mine:
                    self.my_territory.add((x, y))
                else:
                    self.opp_territory.add((x, y))
    
    def update_wall_progress(self, my_eggs: Set[Tuple[int, int]]):
        """Check if wall is completed - now includes both primary and secondary walls."""
        # Primary wall progress
        primary_wall_eggs = self.wall_target_squares & my_eggs
        primary_completion = len(primary_wall_eggs) / max(1, len(self.wall_target_squares))
        
        # Secondary wall progress
        secondary_wall_eggs = self.secondary_wall_squares & my_eggs
        secondary_completion = len(secondary_wall_eggs) / max(1, len(self.secondary_wall_squares))
        
        # Wall is "completed" when primary is 70% done OR secondary is 50% done
        # This gives us flexibility in wall construction
        self.wall_completed = primary_completion >= 0.7 or secondary_completion >= 0.5
        
        return self.wall_completed
    
    def get_wall_priority_score(self, loc: Tuple[int, int], my_eggs: Set[Tuple[int, int]]) -> float:
        """Get priority score for laying egg at this wall location - enhanced for double walls."""
        is_primary = loc in self.wall_target_squares
        is_secondary = loc in self.secondary_wall_squares
        
        if not is_primary and not is_secondary:
            return 0.0
        
        if loc in my_eggs:
            return 0.0
        
        # Primary wall gets HIGHEST priority
        base_priority = 8000.0 if is_primary else 5000.0
        
        # Check for gaps - prioritize filling gaps
        gaps_nearby = 0
        if self.wall_type == 'vertical':
            # Check vertical neighbors for gaps
            for dy in [-2, -1, 1, 2]:
                neighbor = (loc[0], loc[1] + dy)
                if is_primary and neighbor in self.wall_target_squares and neighbor not in my_eggs:
                    gaps_nearby += 1
                elif is_secondary and neighbor in self.secondary_wall_squares and neighbor not in my_eggs:
                    gaps_nearby += 1
        else:
            # Check horizontal neighbors for gaps
            for dx in [-2, -1, 1, 2]:
                neighbor = (loc[0] + dx, loc[1])
                if is_primary and neighbor in self.wall_target_squares and neighbor not in my_eggs:
                    gaps_nearby += 1
                elif is_secondary and neighbor in self.secondary_wall_squares and neighbor not in my_eggs:
                    gaps_nearby += 1
        
        return base_priority + gaps_nearby * 2000.0
    
    def get_all_wall_squares(self) -> Set[Tuple[int, int]]:
        """Get all wall squares (primary + secondary)."""
        return self.wall_target_squares | self.secondary_wall_squares
    
    def is_in_my_territory(self, loc: Tuple[int, int]) -> bool:
        """Check if location is in my territory."""
        return loc in self.my_territory
    
    def is_in_contested_territory(self, loc: Tuple[int, int]) -> bool:
        """Check if location is contested."""
        return loc in self.contested_territory
    
    def should_defend_territory(self, opp_loc: Tuple[int, int], game_phase: float) -> bool:
        """Check if opponent is invading our territory and we should defend."""
        if game_phase > 0.6:  # Late game, focus on eggs not defense
            return False
        
        return opp_loc in self.my_territory


class PlayerAgent:
    def __init__(self, board: Board, time_left: Callable):
        self.tracker = ImprovedTrapdoorTracker()
        self.territory_mgr = None  # Initialize on first turn
        self.transposition_table = TranspositionTable()
        
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.visited_squares = set()
        
        # Game phase tracking
        self.game_initialized = False
        self.current_phase = 'TERRITORY'  # TERRITORY -> SCORING
        
        # Position tracking
        self.position_history = []
        self.HISTORY_LENGTH = 6
        self.last_loc = None
        self.two_back_loc = None
        
        # Timing
        self.start_time = 0
        self.time_limit = 0
        self.N_TOTAL_TURNS = 40
        
        # Caching
        self.territory_cache = {}
        self.cache_generation = 0
        self.nodes_searched = 0
        
        # Parity
        self.my_parity = 0
        self.opp_parity = 0
        
        # Search parameters
        self.last_score = 0
        self.aspiration_window = 200
        self.QUIESCENCE_DEPTH = 3
        self.NULL_MOVE_REDUCTION = 2
        self.LMR_THRESHOLD = 4
        self.LMR_REDUCTION = 1
        
        # ENHANCED EVALUATION WEIGHTS
        # Core scoring
        self.W_EGG_DIFF = 5000.0
        self.W_ABSOLUTE_EGG_COUNT = 800.0
        
        # Territory control (HEAVILY INCREASED)
        self.W_TERRITORY_CONTROL = 2000.0  # NEW: Bonus for being in our territory
        self.W_WALL_EGG = 15000.0  # INCREASED: Wall eggs are critical
        self.W_WALL_COMPLETION = 10000.0  # NEW: Bonus for wall completion
        self.W_TERRITORY_BREACH = -5000.0  # NEW: Penalty if opponent enters our territory
        
        # Risk (DECREASED - be more aggressive)
        self.W_RISK_BASE = 3000.0  # DECREASED from 8000
        self.W_CONFIRMED_DANGER = 50000.0  # NEW: Avoid confirmed traps
        self.W_POTENTIAL_DANGER = 8000.0  # NEW: Moderate avoid potential traps
        
        # Egg opportunities
        self.W_EGG_OPPORTUNITY = 3000.0
        self.W_PATH_TO_EGG = 180.0
        self.W_TERRITORY_UTILIZATION = 400.0
        
        # Mobility and positioning
        self.W_TERRITORY = 250.0
        self.W_MOBILITY = 100.0
        self.W_OPPONENT_MOBILITY = 150.0
        self.W_EXPLORATION = 120.0
        
        # Turds (DE-EMPHASIZED in favor of eggs)
        self.W_CENTRAL_TURD = 400.0  # DECREASED
        self.W_TURD_COUNT_DIFF_BASE = 150.0  # DECREASED
        self.W_EDGE_TURD_PENALTY = 150.0
        self.W_ADJACENT_TURD_PENALTY = 600.0
        
        # Anti-oscillation
        self.W_RETURN_PENALTY = 1500.0
        self.W_OSCILLATION_PENALTY = 60000.0
        
        # Special
        self.W_BLOCK_WIN = 50000.0
        self.W_DIAGONAL_CONTROL = 200.0  # DECREASED
        self.W_CORNER_PROXIMITY = 150.0
        self.CORNER_EGG_BONUS = 3.0
        
        self.TERRITORY_MIN_THRESHOLD = 16
        self.W_TERRITORY_PENALTY = 2500.0
        
        # Game phase thresholds
        self.TERRITORY_PHASE_END = 15  # Turns to spend on territory
        self.LATE_GAME_START = 25  # When to go all-in on eggs

    def _get_game_phase(self, turns_taken: int) -> str:
        """Determine current game phase."""
        if turns_taken < self.TERRITORY_PHASE_END:
            return 'TERRITORY'
        elif turns_taken < self.LATE_GAME_START:
            return 'TRANSITION'
        else:
            return 'SCORING'

    def play(self, board_obj: Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        self.start_time = time()
        self.nodes_searched = 0
        
        # Initialize parity
        self.my_parity = board_obj.chicken_player.even_chicken
        self.opp_parity = board_obj.chicken_enemy.even_chicken
        
        # Update trapdoor tracker
        current_pos = board_obj.chicken_player.get_location()
        turns_taken = self.N_TOTAL_TURNS - board_obj.turns_left_player
        game_phase_num = turns_taken / self.N_TOTAL_TURNS
        
        self.tracker.update(current_pos, sensor_data)
        self.visited_squares.add(current_pos)
        
        # Initialize territory manager on first turn
        if not self.game_initialized:
            self.territory_mgr = TerritoryManager(self.my_parity)
            opp_pos = board_obj.chicken_enemy.get_location()
            self.territory_mgr.initialize_strategy(current_pos, opp_pos, turns_taken)
            self.game_initialized = True
            
            total_wall_squares = len(self.territory_mgr.wall_target_squares) + len(self.territory_mgr.secondary_wall_squares)
            print(f"🗺️  TERRITORY STRATEGY: Claiming {self.territory_mgr.safe_quadrant} side")
            print(f"📊 CLAIMING: {self.territory_mgr.claimed_percentage*100:.0f}% of board ({len(self.territory_mgr.my_territory)} squares)")
            print(f"🧱 WALL SYSTEM: {self.territory_mgr.wall_type} double-wall")
            print(f"🎯 PRIMARY WALL: Line {self.territory_mgr.wall_lines[0]} ({len(self.territory_mgr.wall_target_squares)} eggs)")
            print(f"🛡️  SECONDARY WALL: Line {self.territory_mgr.wall_lines[1]} ({len(self.territory_mgr.secondary_wall_squares)} eggs)")
            print(f"📍 TOTAL WALL EGGS NEEDED: {total_wall_squares}")
        
        # Update game phase
        self.current_phase = self._get_game_phase(turns_taken)
        
        # Check wall completion
        wall_complete = self.territory_mgr.update_wall_progress(set(board_obj.eggs_player))
        
        # Phase-dependent logging
        if self.current_phase == 'TERRITORY':
            primary_eggs = len(self.territory_mgr.wall_target_squares & set(board_obj.eggs_player))
            secondary_eggs = len(self.territory_mgr.secondary_wall_squares & set(board_obj.eggs_player))
            total_wall_eggs = primary_eggs + secondary_eggs
            total_wall_needed = len(self.territory_mgr.wall_target_squares) + len(self.territory_mgr.secondary_wall_squares)
            primary_pct = (primary_eggs / max(1, len(self.territory_mgr.wall_target_squares))) * 100
            secondary_pct = (secondary_eggs / max(1, len(self.territory_mgr.secondary_wall_squares))) * 100
            
            print(f"🧱 TERRITORY PHASE - Wall Progress: {total_wall_eggs}/{total_wall_needed}")
            print(f"   Primary: {primary_eggs}/{len(self.territory_mgr.wall_target_squares)} ({primary_pct:.0f}%) | Secondary: {secondary_eggs}/{len(self.territory_mgr.secondary_wall_squares)} ({secondary_pct:.0f}%)")
        elif self.current_phase == 'TRANSITION' and not wall_complete:
            print(f"⚙️  TRANSITION: Completing walls while scoring")
        else:
            print(f"🥚 SCORING PHASE: Maximize eggs! Current: {board_obj.chicken_player.get_eggs_laid()}")
        
        # Extract state
        state = self._extract_state(board_obj)
        state['game_phase'] = game_phase_num
        state['territory_mgr'] = self.territory_mgr
        state['wall_complete'] = wall_complete
        
        # Check for trapdoor warnings
        risk = self.tracker.get_risk(current_pos, game_phase_num)
        if current_pos in self.tracker.confirmed_danger:
            print(f"⚠️ WARNING: On confirmed danger square {current_pos}!")
        elif current_pos in self.tracker.potential_danger:
            print(f"⚡ CAUTION: On potential danger square {current_pos}")
        
        # Get valid moves
        valid_moves = board_obj.get_valid_moves()
        
        if not valid_moves:
            self.position_history.append(current_pos)
            if len(self.position_history) > self.HISTORY_LENGTH:
                self.position_history.pop(0)
            return (Direction.UP, MoveType.PLAIN)
        
        # CRITICAL: Prioritize EGG moves when on egg square
        can_lay_egg_here = board_obj.can_lay_egg()
        
        if can_lay_egg_here:
            egg_moves = [move for move in valid_moves if move[1] == MoveType.EGG]
            
            if egg_moves:
                print(f"🥚 ON EGG SQUARE - Prioritizing EGG moves")
                # Still include non-egg moves but ordered last
                valid_moves = egg_moves + [m for m in valid_moves if m not in egg_moves]
        
        # Order moves
        valid_moves = self._order_moves(valid_moves, state, board_obj)
        best_move = valid_moves[0]
        
        # Time management
        total_time_rem = time_left()
        turns_rem = board_obj.turns_left_player
        
        if turns_rem > 30:
            base_budget = 3.5
        elif turns_rem > 20:
            base_budget = 5.0
        elif turns_rem > 10:
            base_budget = 7.0
        elif turns_rem > 5:
            base_budget = min(10.0, total_time_rem / max(1, turns_rem) * 0.90)
        else:
            base_budget = min(15.0, total_time_rem / max(1, turns_rem) * 0.95)
        
        safety_buffer = 0.8
        calculated_budget = min(base_budget, total_time_rem - safety_buffer)
        self.turn_budget = max(1.0, calculated_budget)
        self.time_limit = self.start_time + self.turn_budget
        
        # Clear cache periodically
        self.cache_generation += 1
        if self.cache_generation > 5:
            self.territory_cache.clear()
            self.cache_generation = 0
        
        # Iterative deepening search
        depth = 1
        MAX_DEPTH_CAP = 100
        best_depth = 0
        
        try:
            while True:
                if time() - self.start_time > self.turn_budget * 0.92:
                    break
                
                if depth > MAX_DEPTH_CAP:
                    break
                
                # Aspiration window for deeper searches
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
        
        # Safety check: prefer EGG if on egg square
        if best_move and best_move[1] == MoveType.PLAIN and can_lay_egg_here:
            direction = best_move[0]
            egg_alternative = (direction, MoveType.EGG)
            if egg_alternative in valid_moves:
                print(f"🔄 CORRECTING: Forcing EGG move instead of PLAIN")
                best_move = egg_alternative
        
        # Update position history
        new_loc = loc_after_direction(current_pos, best_move[0]) if best_move else current_pos
        self.position_history.append(new_loc)
        if len(self.position_history) > self.HISTORY_LENGTH:
            self.position_history.pop(0)
        
        self.two_back_loc = self.last_loc
        self.last_loc = new_loc
        
        # Logging
        tt_hit_rate = 0
        if self.transposition_table.hits + self.transposition_table.misses > 0:
            tt_hit_rate = self.transposition_table.hits / (self.transposition_table.hits + self.transposition_table.misses)
        
        eggs_laid = board_obj.chicken_player.get_eggs_laid()
        print(f"⏱️ TIME: {time() - self.start_time:.3f}s | Depth: {best_depth} | Eggs: {eggs_laid} | TT: {tt_hit_rate:.2%}")
        
        return best_move

    def _order_moves(self, moves: List[Tuple[Direction, MoveType]], state: Dict, board_obj: Board) -> List[Tuple[Direction, MoveType]]:
        """Enhanced move ordering with territory awareness."""
        current_loc = state['my_loc']
        game_phase_num = state['game_phase']
        territory_mgr = state['territory_mgr']
        wall_complete = state['wall_complete']
        
        on_egg_square = (current_loc[0] + current_loc[1]) % 2 == self.my_parity
        
        def move_priority(move):
            score = 0
            direction, move_type = move
            new_loc = loc_after_direction(current_loc, direction)
            
            # Killer move bonus
            if move in self.killer_moves.get(0, []):
                score += 10000
            
            score += self.history_table.get(move, 0)
            
            # PHASE-BASED PRIORITIES
            if self.current_phase == 'TERRITORY':
                # Priority 1: Build BOTH walls
                if move_type == MoveType.EGG:
                    if on_egg_square and current_loc in territory_mgr.wall_target_squares:
                        score += 150000  # PRIMARY WALL - ABSOLUTE PRIORITY
                        print(f"🧱🔴 PRIMARY WALL EGG at {current_loc}!")
                    elif on_egg_square and current_loc in territory_mgr.secondary_wall_squares:
                        score += 120000  # SECONDARY WALL - VERY HIGH PRIORITY
                        print(f"🧱🟡 SECONDARY WALL EGG at {current_loc}!")
                    elif on_egg_square and territory_mgr.is_in_my_territory(current_loc):
                        score += 60000  # Interior territory eggs - good for control
                    elif on_egg_square:
                        score += 25000  # Any egg is still decent
                
                # Priority 2: Move toward NEAREST wall square (primary or secondary)
                if move_type == MoveType.PLAIN:
                    all_wall_squares = territory_mgr.get_all_wall_squares()
                    incomplete_walls = [sq for sq in all_wall_squares if sq not in state['my_eggs']]
                    
                    if incomplete_walls:
                        wall_distances = [manhattan_distance(new_loc, sq) for sq in incomplete_walls]
                        min_dist = min(wall_distances)
                        score -= min_dist * 1200  # Strong pull toward wall squares
                
                # Priority 3: Strategic turds on/near wall lines for defense
                if move_type == MoveType.TURD:
                    if current_loc in territory_mgr.wall_target_squares:
                        score += 15000  # Turds on primary wall line
                    elif current_loc in territory_mgr.secondary_wall_squares:
                        score += 12000  # Turds on secondary wall line
                    elif territory_mgr.is_in_my_territory(current_loc):
                        # Turds in our territory for extra defense
                        score += 8000
            
            elif self.current_phase == 'TRANSITION':
                # Finish walls aggressively while starting to score
                if move_type == MoveType.EGG:
                    if on_egg_square and current_loc in territory_mgr.wall_target_squares and not wall_complete:
                        score += 130000  # Still prioritize primary wall
                    elif on_egg_square and current_loc in territory_mgr.secondary_wall_squares and not wall_complete:
                        score += 100000  # Still prioritize secondary wall
                    elif on_egg_square:
                        score += 70000  # But also maximize eggs everywhere
                
                if move_type == MoveType.PLAIN:
                    # Balance: move toward walls if incomplete, otherwise toward any egg square
                    if not wall_complete:
                        all_wall_squares = territory_mgr.get_all_wall_squares()
                        incomplete_walls = [sq for sq in all_wall_squares if sq not in state['my_eggs']]
                        if incomplete_walls:
                            wall_distances = [manhattan_distance(new_loc, sq) for sq in incomplete_walls]
                            score -= min(wall_distances) * 600
                    else:
                        # Just move toward egg squares
                        if (new_loc[0] + new_loc[1]) % 2 == self.my_parity:
                            score += 8000
            
            else:  # SCORING phase
                # All-in on eggs - walls should be done
                if move_type == MoveType.EGG and on_egg_square:
                    score += 120000  # Massive egg priority
                elif move_type == MoveType.EGG:
                    score += 60000
                
                # Move toward egg squares
                if move_type == MoveType.PLAIN:
                    if (new_loc[0] + new_loc[1]) % 2 == self.my_parity:
                        score += 15000  # Strong pull to egg squares
            
            # TRAPDOOR RISK (Enhanced)
            risk = self.tracker.get_risk(new_loc, game_phase_num)
            
            if new_loc in self.tracker.confirmed_danger:
                score -= 100000  # Never go here
            elif new_loc in self.tracker.potential_danger:
                score -= 20000  # Strongly avoid
            elif new_loc in self.tracker.confirmed_safe:
                score += 2000  # Prefer safe squares
            else:
                score -= risk * 5000  # Moderate risk penalty
            
            # Territory bonus
            if territory_mgr.is_in_my_territory(new_loc):
                score += 1000
            
            # Mobility
            potential_moves = self._count_potential_moves(new_loc, state, is_me=True)
            score += potential_moves * 50
            
            # Anti-oscillation
            if new_loc in self.position_history[-4:] and move_type != MoveType.EGG:
                score -= 500
            
            return score
        
        return sorted(moves, key=move_priority, reverse=True)

    def evaluate(self, state: Dict) -> float:
        """Enhanced evaluation with territory control."""
        score = 0.0
        
        # Basic egg differential
        egg_diff = state['my_score'] - state['opp_score']
        score += egg_diff * self.W_EGG_DIFF
        
        # Absolute egg count
        score += state['my_score'] * self.W_ABSOLUTE_EGG_COUNT
        
        # Game phase
        game_phase = state['game_phase']
        territory_mgr = state['territory_mgr']
        wall_complete = state.get('wall_complete', False)
        
        # TERRITORY CONTROL EVALUATION
        if self.current_phase == 'TERRITORY':
            # Wall completion bonus - BOTH walls matter
            primary_wall_eggs = len(territory_mgr.wall_target_squares & state['my_eggs'])
            primary_ratio = primary_wall_eggs / max(1, len(territory_mgr.wall_target_squares))
            
            secondary_wall_eggs = len(territory_mgr.secondary_wall_squares & state['my_eggs'])
            secondary_ratio = secondary_wall_eggs / max(1, len(territory_mgr.secondary_wall_squares))
            
            # Higher bonus for primary wall
            score += primary_ratio * self.W_WALL_COMPLETION * 1.5
            score += secondary_ratio * self.W_WALL_COMPLETION
            
            # Penalty if opponent in our territory
            if territory_mgr.is_in_my_territory(state['opp_loc']):
                score += self.W_TERRITORY_BREACH
            
            # Bonus for being in our territory
            if territory_mgr.is_in_my_territory(state['my_loc']):
                score += self.W_TERRITORY_CONTROL
        
        # Wall egg bonus (always) - count BOTH walls
        all_wall_squares = territory_mgr.get_all_wall_squares()
        total_wall_eggs = len(all_wall_squares & state['my_eggs'])
        wall_multiplier = 2.0 if not wall_complete else 0.8
        score += total_wall_eggs * self.W_WALL_EGG * wall_multiplier
        
        # TRAPDOOR RISK (Enhanced)
        my_risk = self.tracker.get_risk(state['my_loc'], game_phase)
        
        if state['my_loc'] in self.tracker.confirmed_danger:
            score -= self.W_CONFIRMED_DANGER
        elif state['my_loc'] in self.tracker.potential_danger:
            score -= self.W_POTENTIAL_DANGER
        else:
            score -= my_risk * self.W_RISK_BASE
        
        # Egg opportunities
        available_eggs = self._count_available_egg_squares(state, is_me=True)
        opp_available = self._count_available_egg_squares(state, is_me=False)
        score += (available_eggs - opp_available) * 200
        
        # Territory control (flood fill)
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
        
        # Mobility
        my_mobility = self._count_potential_moves(state['my_loc'], state, is_me=True)
        opp_mobility = self._count_potential_moves(state['opp_loc'], state, is_me=False)
        score += (my_mobility - opp_mobility) * self.W_MOBILITY
        score -= opp_mobility * self.W_OPPONENT_MOBILITY
        
        # Path to egg square
        if (state['my_loc'][0] + state['my_loc'][1]) % 2 != self.my_parity:
            nearest_egg = self._find_nearest_egg_square(state['my_loc'], state)
            if nearest_egg:
                dist = manhattan_distance(state['my_loc'], nearest_egg)
                score -= dist * self.W_PATH_TO_EGG * (1 + game_phase)
        
        # Turd positioning
        central_turds = sum(1 for loc in state['my_turds'] if 2 <= loc[0] <= 5 and 2 <= loc[1] <= 5)
        score += central_turds * self.W_CENTRAL_TURD
        
        edge_turds = sum(1 for loc in state['my_turds'] if loc[0] in [0, 7] or loc[1] in [0, 7])
        score -= edge_turds * self.W_EDGE_TURD_PENALTY
        
        # Anti-oscillation
        current_loc = state['my_loc']
        last_loc = state.get('last_loc')
        two_back = state.get('two_back_loc')
        
        if last_loc and current_loc == last_loc:
            score -= self.W_RETURN_PENALTY * 2
        if two_back and current_loc == two_back:
            score -= self.W_OSCILLATION_PENALTY
        
        return score

    def _count_available_egg_squares(self, state: Dict, is_me: bool) -> int:
        """Count available egg-laying squares."""
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

    def _find_nearest_egg_square(self, loc: Tuple[int, int], state: Dict) -> Optional[Tuple[int, int]]:
        """Find nearest available egg square."""
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
        """Count potential moves from a position."""
        count = 0
        other_loc = state['opp_loc'] if is_me else state['my_loc']
        opp_eggs = state['opp_eggs'] if is_me else state['my_eggs']
        opp_turds = state['opp_turds'] if is_me else state['my_turds']
        
        for d in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            new_loc = loc_after_direction(loc, d)
            nx, ny = new_loc
            
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue
            if new_loc == other_loc:
                continue
            if new_loc in opp_eggs or new_loc in opp_turds:
                continue
            
            # Check turd zones
            in_turd_zone = False
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                if (nx + dx, ny + dy) in opp_turds:
                    in_turd_zone = True
                    break
            
            if not in_turd_zone:
                count += 1
        
        return count

    def _flood_fill_count(self, state: Dict, is_me_perspective: bool) -> int:
        """Count reachable territory using flood fill."""
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
                if new_loc in obstacles or new_loc in enemy_turds:
                    continue
                
                # Check turd zones
                blocked = False
                for tdx, tdy in deltas:
                    if (nx + tdx, ny + tdy) in enemy_turds:
                        blocked = True
                        break
                
                if not blocked:
                    visited.add(new_loc)
                    queue.append(new_loc)
        
        return len(visited)

    def _extract_state(self, board_obj: Board) -> Dict:
        """Extract game state."""
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

    def minimax(self, state, depth, alpha, beta, maximizing):
        """Minimax with alpha-beta pruning."""
        self.nodes_searched += 1
        
        if time() > self.time_limit:
            raise TimeoutError()
        
        # Transposition table lookup
        tt_value, tt_move = self.transposition_table.get(state, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move
        
        # Terminal depth
        if depth == 0:
            return self.evaluate(state), None
        
        # Get moves
        moves = self._get_valid_moves_sim(state, is_me=maximizing)
        
        if not moves:
            terminal_eval = self.evaluate_terminal(state, maximizing)
            self.transposition_table.store(state, depth, terminal_eval, 'EXACT', None)
            return terminal_eval, None
        
        # Order moves
        moves = self._order_moves_sim(moves, state, maximizing)
        best_move = moves[0]
        
        if maximizing:
            max_eval = float('-inf')
            
            for move in moves:
                new_state = self._apply_move_sim(state, move, is_me=True)
                eval_val, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    # Killer move
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop()
                    break
            
            flag = 'EXACT' if max_eval > alpha and max_eval < beta else ('LOWERBOUND' if max_eval >= beta else 'UPPERBOUND')
            self.transposition_table.store(state, depth, max_eval, flag, best_move)
            return max_eval, best_move
        
        else:
            min_eval = float('inf')
            
            for move in moves:
                new_state = self._apply_move_sim(state, move, is_me=False)
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
            
            flag = 'EXACT' if min_eval > alpha and min_eval < beta else ('LOWERBOUND' if min_eval >= beta else 'UPPERBOUND')
            self.transposition_table.store(state, depth, min_eval, flag, best_move)
            return min_eval, best_move

    def evaluate_terminal(self, state, am_i_blocked):
        """Evaluate terminal positions."""
        base = self.evaluate(state)
        return base - self.W_BLOCK_WIN if am_i_blocked else base + self.W_BLOCK_WIN

    def _order_moves_sim(self, moves: List[Tuple[Direction, MoveType]], state: Dict, is_me: bool) -> List[Tuple[Direction, MoveType]]:
        """Fast move ordering for simulation."""
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
            
            return score
        
        return sorted(moves, key=priority, reverse=True)

    def _apply_move_sim(self, state: Dict, move: Tuple[Direction, MoveType], is_me: bool) -> Dict:
        """Apply move in simulation."""
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
        """Get valid moves in simulation."""
        moves = []
        curr_loc = state['my_loc'] if is_me else state['opp_loc']
        other_loc = state['opp_loc'] if is_me else state['my_loc']
        turds_left = state['my_turds_left'] if is_me else state['opp_turds_left']
        required_parity = self.my_parity if is_me else self.opp_parity
        
        my_eggs = state['my_eggs'] if is_me else state['opp_eggs']
        opp_eggs = state['opp_eggs'] if is_me else state['my_eggs']
        my_turds = state['my_turds'] if is_me else state['opp_turds']
        opp_turds = state['opp_turds'] if is_me else state['my_turds']
        
        cell_parity = (curr_loc[0] + curr_loc[1]) % 2
        dist_to_enemy = manhattan_distance(curr_loc, other_loc)
        
        is_curr_empty = (curr_loc not in my_eggs and curr_loc not in opp_eggs and 
                        curr_loc not in my_turds and curr_loc not in opp_turds)
        
        can_egg = (cell_parity == required_parity and is_curr_empty and dist_to_enemy > 1)
        can_turd = (turds_left > 0 and is_curr_empty and dist_to_enemy > 1)
        
        dirs = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
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
            
            # Check turd zones
            in_turd_zone = False
            for tdx, tdy in deltas:
                if (nx + tdx, ny + tdy) in opp_turds:
                    in_turd_zone = True
                    break
            
            if in_turd_zone:
                continue
            
            moves.append((d, MoveType.PLAIN))
            if can_egg:
                moves.append((d, MoveType.EGG))
            if can_turd:
                moves.append((d, MoveType.TURD))
        
        return moves