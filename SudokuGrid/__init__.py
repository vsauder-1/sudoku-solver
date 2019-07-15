"""Sudoku Game representations"""
import collections
import copy
import itertools
import random

DEBUG = True
SHOW_EACH_MOVE = True
SHUFFLE_RULES = True

CHOICES = list(range(1, 10))
ROW_NAMES = CHOICES
COL_NAMES = CHOICES


def dbg(fmt, *args, **kwargs):
    if DEBUG:
        print(fmt.format(*args, **kwargs))


def dprint(*args):
    if DEBUG:
        print(*args)


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def cross(A, B):
    """Cross product of elements in A and elements in B."""
    return [(a, b) for a in A for b in B]


class Cell:
    def __init__(self, addr):
        self.possible = CHOICES.copy()
        self._given = None
        self.name = addr

    @property
    def row(self):
        return self.name[0]

    @property
    def col(self):
        return self.name[1]

    @property
    def given(self):
        return self._given

    @given.setter
    def given(self, v):
        self._given = v
        self.possible = [v]

    @property
    def solved(self):
        return len(self.possible) == 1

    def __eq__(self, other):
        return self.name == other.name and self.possible == other.possible

    def remove(self, v):
        # dbg('Trying to remove {} from {}, exist: {}', v, self.name, self.possible)
        if v in self.possible:
            self.possible.remove(v)

    def solve(self, v):
        self.possible = [v]

    def __copy__(self):
        n = Cell()
        n._given = self._given
        n.possible = self.possible

    def __str__(self):
        hdr = f'Cell {self.name}: '
        if self._given:
            return hdr + f'given: {self._given}'
        return hdr + f'possible: {self.possible_str}'

    @property
    def possible_str(self):
        return ''.join([str(n) for n in self.possible])


class Game:
    RCLEN = len(CHOICES)
    GRIDLEN = RCLEN ** 2

    def __init__(self, values=None):
        if values and len(values) != self.GRIDLEN:
            raise ValueError(f"Grid requires exactly {self.GRIDLEN} values for initialization.")
        cellidx = itertools.chain.from_iterable(cross([r], COL_NAMES) for r in ROW_NAMES)
        self._cellDict = collections.OrderedDict([(idx, Cell(idx)) for idx in cellidx])
        for v, k in zip(values or [], self._cellDict.keys()):
            try:
                if int(v) in CHOICES:
                    self._cellDict[k].given = int(v)
            except (ValueError, TypeError):
                pass
        self._define_regions()
        self._rules = []

    def at(self, r, c):
        return self._cellDict[(r, c)]

    @property
    def rows(self):
        return (self.get_row(r) for r in ROW_NAMES)

    @property
    def cols(self):
        return (self.get_col(r) for r in COL_NAMES)

    def get_row(self, row):
        return [self._cellDict[i] for i in cross([row], COL_NAMES)]

    def get_col(self, col):
        return [self._cellDict[i] for i in cross(ROW_NAMES, [col])]

    def _define_regions(self):
        cols = [cross(ROW_NAMES, [c]) for c in COL_NAMES]
        rows = [cross([r], COL_NAMES) for r in ROW_NAMES]
        self._regions = cols + rows

    def display(self):
        print(self.format())

    def format(self):
        width = 1 + max(len(s.possible) for s in self._cellDict.values())
        dline = '+'.join(['-' * (width * 3 + 1)] * 3)

        def fmtline(lin):
            strs = [''.join(v.possible_str) for v in lin]
            return ' ' + ''.join(c.center(width)+('| ' if i in [2, 5] else '')
                           for i, c in enumerate(strs))

        def gen():
            for r in ROW_NAMES:
                yield fmtline(self.get_row(r))
                if r in [3, 6]:
                    yield dline
        return '\n'.join(gen())

    def text(self):
        return ''.join([(c.possible_str if c.solved else '.') for c in self._cellDict.values()])

    def is_solved(self):
        return all(c.solved for c in self._cellDict.values())

    def verify_solution(self):
        for R in self.region_cells:
            for D in CHOICES:
                exist = [c for c in R if D in c.possible]
                if len(exist) != 1:
                    print(f'Value {D} must exist in at least one cell of {[str(a) for a in R]}')
                    return False
        return True

    def solve(self, display=True):
        for t in range(99):
            rules = self._rules
            if SHUFFLE_RULES:
                rules = rules.copy()
                random.shuffle(rules)
            orig = copy.deepcopy(self._cellDict)
            if display and SHOW_EACH_MOVE:
                prerule = copy.deepcopy(self._cellDict)
                for r in rules:
                    r()
                    if prerule != self._cellDict:
                        print('<' * 40, 'Changed by', r.__name__, '>' * 40)
                        self.display()
                        prerule = copy.deepcopy(self._cellDict)
            else:
                for r in rules:
                    r()
            if orig == self._cellDict:
                if display:
                    print('Unsolvable by rules')
                break

            if display:
                if SHOW_EACH_MOVE:
                    print('Completed round', t)
                else:
                    print('<' * 40, f'After round {t}', '>' * 40)
                    self.display()
            if self.is_solved():
                self.verify_solution()
                break
        else:
            print('Cannot solve this puzzle')

    def brute(self, display=True):
        orig = copy.deepcopy(self._cellDict)
        cell2s = [c for c in orig.values() if len(c.possible) == 2]
        for c2 in cell2s:
            # try the possible values
            for n in c2.possible:
                if display:
                    print(f'Trying value {n} of {c2.possible_str} in {c2.name}')
                self._cellDict[c2.name].solve(n)
                try:
                    self.solve(display=display)
                except (AssertionError, ValueError) as e:
                    if display:
                        print('Brute solve failed with contraction:', e)
                        self.display()
                if self.is_solved():
                    return
                self._cellDict = orig
        if display:
            print('Cannot solve this puzzle by brute force')

    def peers(self, rc):
        reg_idc = itertools.chain.from_iterable(r for r in self._regions if rc in r)
        return [self._cellDict[c] for c in reg_idc if c != rc]

    @staticmethod
    def open_cells(cells, digit=None):
        if digit:
            return [c for c in cells if digit in c.possible]
        return [c for c in cells]

    @staticmethod
    def possibilities(cells):
        return [c.possible for c in cells if not c.solved]

    @property
    def region_cells(self):
        return [[self._cellDict[rc] for rc in reg] for reg in self._regions]

    @property
    def regions(self):
        return self._regions

    def _apply_rules(self):
        for r in self._rules:
            r()


class SquareGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rules += [self.peer_rule, self.one_per_region, self.pairex_rule, self.pair_rule, self.tripex_rule,
                        self.band_single_row_rule, self.stack_single_col_rule,
                        self.x_wing_row_rule, self.x_wing_col_rule]

    def peer_rule(self):
        for v in self._cellDict.values():
            if v.solved:
                for p in self.peers(v.name):
                    if not v.possible:
                        self.display()
                        raise ValueError(f'Exhausted possibilities for {v}')
                    p.remove(v.possible[0])

    def one_per_region(self):
        for R in self.region_cells:
            for D in CHOICES:
                exist = [c for c in R if D in c.possible]
                if not exist:
                    raise ValueError(f'Value {D} must exist in at least one cell of {[str(a) for a in R]}')
                if len(exist) == 1:
                    exist[0].solve(D)

    def pair_rule(self):
        for R in self.region_cells:
            all_poss = [c.possible for c in R]
            pairs = [p for p in all_poss if len(p) == 2]
            for np in [p for p in pairs if pairs.count(p) == 2]:
                for n in np:
                    for c in R:
                        if c.possible != np:
                            c.remove(n)

    def pairex_rule(self):
        def segregate(cells, pred):
            inl, outl = list(), list()
            for c in cells:
                if pred(c):
                    inl.append(c)
                else:
                    outl.append(c)
            return inl, outl
        for R in self.region_cells:
            all_poss = [c.possible for c in R if not c.solved]
            totposs = set(itertools.chain.from_iterable(all_poss))
            for pp in itertools.combinations(totposs, 2):
                pps = set(pp)
                ex, notex = segregate(R, lambda c: pps.issubset(c.possible))
                if len(ex) == 2 and not [c for c in notex if pps.intersection(c.possible)]:
                    # dprint('Found exclusive pair: ', pp, [str(c) for c in ex])
                    for c in ex:
                        for n in set(totposs) - pps:
                            c.remove(n)

    def tripex_rule(self):
        def segregate(cells, pred):
            inl, outl = list(), list()
            for c in cells:
                if pred(c):
                    inl.append(c)
                else:
                    outl.append(c)
            return inl, outl
        for R in self.region_cells:
            all_poss = [c.possible for c in R if not c.solved]
            totposs = set(itertools.chain.from_iterable(all_poss))
            for tt in [set(t) for t in itertools.combinations(totposs, 3)]:
                ex, notex = segregate(R, lambda c: tt.issubset(c.possible))
                if len(ex) == 3 and not [c for c in notex if tt.intersection(c.possible)]:
                    # dprint('Found exclusive triplet: ', tt, [str(c) for c in ex])
                    for c in ex:
                        for n in set(totposs) - tt:
                            c.remove(n)

    def band_single_row_rule(self):
        """If only one row can hold a number, the cells in the same row but other regions in the band get cleared"""
        for rows, boxes in self.bands:
            rows = list(rows)
            for b in boxes:
                for ch in set(itertools.chain.from_iterable([c.possible for c in b if not c.solved])):
                    cells = [c for c in b if ch in c.possible]
                    for r in rows:
                        if all([c in r for c in cells]):
                            for c in [cr for cr in r if not cr.solved and cr not in b]:
                                c.remove(ch)

    def stack_single_col_rule(self):
        """If only one column can hold a number, the cells in the same column but other regions in the stack get cleared"""
        for cols, boxes in self.stacks:
            cols = list(cols)
            for b in boxes:
                # find the set of all possible choices for the box
                for ch in set(itertools.chain.from_iterable([c.possible for c in b if not c.solved])):
                    # list the cells where the choice is present
                    cells = [c for c in b if ch in c.possible]
                    for col in cols:
                        # if all the cells that can have the choice are in this column,
                        if all([c in col for c in cells]):
                            # remove the choice from the rest of the cells of this column that are not in this box
                            for c in [cc for cc in col if not cc.solved and cc not in b]:
                                c.remove(ch)

    def x_wing_row_rule(self):
        """Solve case where 1 digit is present as only option in 2 matching columns for 2 different rows

         https://www.livesudoku.com/en/tutorial-x-wing.php"""
        for D in CHOICES:
            xcells = collections.defaultdict(list)
            # for the rows where the choice is exclusive to 2 cells; save the row/columns
            for cells in (self.open_cells(R, D) for R in self.rows):
                if len(cells) == 2:
                    xcells[tuple(c.col for c in cells)].append(cells[0].row)
            # for each case where only 2 columns have the choice possible in 2 exclusive rows
            for cols, xrow in xcells.items():
                if len(xrow) == 2:
                    # remove the choice from all cells in these columns except for the exclusive cols
                    others = list(c for C in cols for c in self.get_col(C) if D in c.possible and c.row not in xrow)
                    if others:
                        dprint(f'Found X-WING at columns {cols} for {D}, exclusive rows are {xrow}')
                        for c in others:
                            c.remove(D)

    def x_wing_col_rule(self):
        """Solve case where 1 digit is present as only option in 2 matching rows for 2 different columns

         https://www.livesudoku.com/en/tutorial-x-wing.php"""
        for D in CHOICES:
            xcells = collections.defaultdict(list)
            # for the columns where the choice is exclusive to 2 cells; save the rows/column
            for cells in (self.open_cells(C, D) for C in self.cols):
                if len(cells) == 2:
                    xcells[tuple(c.row for c in cells)].append(cells[0].col)
            # for each case where only 2 rows have the choice possible in 2 exclusive columns
            for rows, xcol in xcells.items():
                if len(xcol) == 2:
                    # remove the choice from all cells in these rows except for the exclusive columns
                    others = list(c for R in rows for c in self.get_row(R) if D in c.possible and c.col not in xcol)
                    if others:
                        dprint(f'Found X-WING at rows {rows} for {D}, exclusive columns are {xcol}')
                        for c in others:
                            c.remove(D)

    def swordfish_row_rule(self):
        """Solve case where 1 digit is present as option in only 3 matching columns for 3 different rows

         https://www.livesudoku.com/en/tutorial-swordfish.php"""
        for D in CHOICES:
            xcells = collections.defaultdict(set)
            # for the rows where the choice is exclusive to 2 cells; save the row/columns
            for cells in (self.open_cells(R, D) for R in self.rows):
                if len(cells) <= 3:
                    xcells[tuple(c.col for c in cells)] |= cells[0].row
            # for each case where only 2 columns have the choice possible in 2 exclusive rows
            for cols, xrow in xcells.items():
                if len(xrow) == 2:
                    # remove the choice from all cells in these columns except for the exclusive cols
                    others = list(c for C in cols for c in self.get_col(C) if D in c.possible and c.row not in xrow)
                    if others:
                        dprint(f'Found X-WING at columns {cols} for {D}, exclusive rows are {xrow}')
                        for c in others:
                            c.remove(D)

    def _define_regions(self):
        super()._define_regions()
        boxes = [cross(rs, cs) for rs in grouper(ROW_NAMES, 3) for cs in grouper(COL_NAMES, 3)]
        self._regions += boxes
        self._bands = [(rs, bxs) for rs, bxs in zip(grouper(ROW_NAMES, 3), grouper(boxes, 3))]
        self._stacks = [(cs, bxs) for cs, bxs in zip(grouper(COL_NAMES, 3), [itertools.islice(boxes, n, None, 3) for n in range(3)])]

    def get_band(self, bdx):
        b = self._bands[bdx]
        return (self.get_row(n) for n in b[0]), [[self._cellDict[rc] for rc in reg] for reg in b[1]]

    def get_stack(self, bdx):
        b = self._stacks[bdx]
        return (self.get_col(n) for n in b[0]), [[self._cellDict[rc] for rc in reg] for reg in b[1]]

    @property
    def bands(self):
        return (self.get_band(b) for b in range(3))

    @property
    def stacks(self):
        return (self.get_stack(b) for b in range(3))


class Hyper(SquareGame):
    def _define_regions(self):
        super()._define_regions()
        self._regions += [cross(rs, cs) for rs in [range(1, 4), range(5, 8)] for cs in [range(1, 4), range(5, 8)]]
