import pathlib
import time

from SudokuGrid import *

SuperEasy = '.8.4.96536428...7.......8....7..5.42...7.1...85.6..1....6.......1...47362735.8.1.'
grid1 = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
Easy45 = pathlib.Path('Printable45.txt').read_text().splitlines()
Easy40 = pathlib.Path('Printable40.txt').read_text().splitlines()
my_unsolved = '''\
631247859478695213259813467.14..269892718653486.9.472114..68972.86.29145.924.1386
7569821343294176..41836597283..96741.74.31.9..917482.394.1.3...1836.942..678.4319
589.16..426..7...8..4.8.....25.618.914.892..569853.2.143.12958.8...5.492952.481..
49378.56.716..3.84528.467.3871465329269378145345...6789.7.248.66.48.7...182639457
3.895..4.94..6358.567814329495138.7.6..47589378362915427938146583654.91.154.96.38
42..8961591.4.5823..52.174925.8.6471....145.21.45.23.6531948267798623154642157938
56.8...1.9.1.65.8.8.2..1.56615...8473987546217246185392591..4684..586192186429375
9638..5717..315689815967324..81..763671..39483..678152.8.7.123..3..8.41.1..43689.
83524.76.4763.....921876..42541.7.9679362.1..16859.2.76879..4..54276.9..3194.267.
19..82743..8.74591..4.19682951238467.6.451829482796135849125376...863914.1.947258
'''.splitlines()

unsolved_hyper = ['..1.6..9886279.1.3594.1.726.5.1..964..965.38.6.........26...87...5..6.399.....6..']
fiendish_hyper80 = '.1..3..2..............7..51.6.7..1.5.........9.1..4.3.34..5..............9..8..1.'
fiendish1 =  '...5...6.8.9....1.16..87...3...26.....7.1.6.....85...3...47..21.4....9.8.8...3...'
fiendish60 = '......5......6..1.9542....6.3..976.............183..7.2....4935.8..1......6......'

DISPLAY = True
BRUTE = False

def main():
    solved = 0
    solvtime = []
    brutetime = []

    for e, t in enumerate([fiendish60]):
    # for e, t in enumerate(unsolved_hyper):
        if not t:
            continue
        # g = Hyper(t)
        g = SquareGame(t)
        if DISPLAY:
            print('.=' * 40)
            print(f'Game #{e}')
            g.display()
        stime = time.monotonic()
        g.solve(display=DISPLAY)
        if g.is_solved():
            solvtime.append(time.monotonic() - stime)
            solved += 1
        else:
            if BRUTE:
                print('Trying to solve by brute force')
                btime = time.monotonic()
                g.brute(display=DISPLAY)
            if g.is_solved():
                brutetime.append(time.monotonic() - btime)
                solvtime.append(time.monotonic() - stime)
                solved += 1
            else:
                print(f'failed to solve #{e}')
                g.display()
                print(g.text())
    print(f'Solved {solved} of {e+1}')
    if solved:
        print(f'Times: total:{sum(solvtime):.3f} avg:{sum(solvtime)/(len(solvtime)):.3f} max:{max(solvtime):.3f}')
    if brutetime:
        print(f'Brute times: total:{sum(brutetime):.3f} avg:{sum(brutetime)/(len(brutetime)):.3f} max:{max(brutetime):.3f}')

if __name__ == '__main__':
    main()