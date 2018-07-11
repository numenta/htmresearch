from htmresearch.frameworks.specific_timing.sequenceLearner import DoubleADTM


Sequence1 = (('A', 5), ('B', 8), ('C', 12), ('D', 16))
Sequence2 = (('A', 10), ('B', 11), ('C', 14), ('D', 16))
Sequence3 = (('X', 5), ('B', 8), ('C', 12), ('Y', 16))
Sequence4 = (('A', 4), ('B', 8), ('C', 12), ('D', 16))
Sequence5 = (('A', 5), ('X', 8), ('C', 12), ('D', 16))


def test_1(seq_1, seq_2, n_iter):
    d_adtm = DoubleADTM(num_columns=2048,
                        num_active_cells=39,
                        num_time_columns=1024,
                        num_active_time_cells=19,
                        num_time_steps=20)

    for _ in range(n_iter):

        d_adtm.learn(train_seq=seq_1, num_iter=1)
        d_adtm.learn(train_seq=seq_2, num_iter=1)

    d_adtm.infer(test_seq=seq_1)
    d_adtm.infer(test_seq=seq_2)


def test_2(seq_1, seq_2):
    d_adtm = DoubleADTM(num_columns=2048,
                        num_active_cells=39,
                        num_time_columns=1024,
                        num_active_time_cells=19,
                        num_time_steps=20)

    d_adtm.learn(train_seq=seq_1, num_iter=20)

    d_adtm.infer(test_seq=seq_1)
    d_adtm.infer(test_seq=seq_2)


'test 1a'
test_1(Sequence1, Sequence2, n_iter=7)

'test 1b'
test_1(Sequence1, Sequence3, n_iter=7)

'test 2a'
test_2(Sequence1, Sequence4)

'test 2b'
test_2(Sequence1, Sequence5)

