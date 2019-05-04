from grammar_algorithms.Sequitur import run_sequitur

def test_Sequitur():
    assert run_sequitur(
        'abracadabraabracadabra') == 'Usage\tRule\n 0\tR0 -> R1 R1 \n 2\tR1 -> R2 c a d R2 \n 2\tR2 -> a b r a \n'
    assert run_sequitur('11111211111') == 'Usage\tRule\n 0\tR0 -> R1 R2 2 R2 R1 \n 3\tR1 -> 1 1 \n 2\tR2 -> R1 1 \n'

    print(run_sequitur('1111'))
    print('Usage\tRule\n 0\tR0 -> R1 R1 \n 2\tR1 -> 1 1 \n')


    assert run_sequitur('1111') == 'Usage\tRule\n 0\tR0 -> R1 R1 \n 2\tR1 -> 1 1\n'