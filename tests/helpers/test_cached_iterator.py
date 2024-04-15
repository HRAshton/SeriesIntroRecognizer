import pytest

from series_intro_recognizer.helpers.cached_iterator import iterate_with_cache


def test_returns_empty_list():
    test_list = []
    test_iter = iter(test_list)

    result = list(iterate_with_cache(test_iter, 2))

    expected_result = []
    assert result == expected_result


def test_returns_single_item():
    test_list = [1]
    test_iter = iter(test_list)

    result = list(iterate_with_cache(test_iter, 2))

    assert result == []


def test_attaches_index():
    test_list = [100, 200, 300, 400, 500, 600]
    test_iter = iter(test_list)

    result = list(iterate_with_cache(test_iter, 2))

    for a, b in result:
        assert a[1] == (a[0] + 1) * 100
        assert b[1] == (b[0] + 1) * 100


@pytest.mark.parametrize("n", [2, 3, 4])
def test_returns_pairs(n):
    test_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    test_iter = iter(test_list)

    result = list(iterate_with_cache(test_iter, n))
    print(result)

    # Check that 100 is attached to 200, 300, 400 and so on
    for i in range(len(test_list) - 1):
        for j in range(i + 1, min(i + n + 1, len(test_list))):
            next_item = result.pop(0)
            assert next_item[0] == (i, test_list[i])
            assert next_item[1] == (j, test_list[j])


@pytest.mark.parametrize("block_size", [2, 3, 20])
def test_invoke_iter_on_demand(block_size):
    cnt = 0

    def get_next():
        for _ in range(100):
            nonlocal cnt
            cnt += 1
            yield 0

    result = iterate_with_cache(get_next(), block_size)

    # First call should consume block_size elements
    # Every Nnd subsequent call should consume one element
    for i in range(10):
        for _ in range(block_size):
            print(next(result), cnt)
            assert cnt == block_size + i + 1
