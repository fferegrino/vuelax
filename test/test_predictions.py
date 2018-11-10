from predictions import create_trip
import pytest


def create_response(origin, destination, price):
    return {
        'origin': origin,
        'destination': destination,
        'price': price
    }


t1 = ['¡', 'CDMX', 'y', 'MTY', 'a', 'Japón', '1000']
l1 = ['n', 'o', 'o', 'o', 's', 'd', 'p']
res1 = create_response('CDMX y MTY', 'Japón', 1000)

t2 = ['¡', 'CDMX', ',', 'MTY', 'a', 'Japón', '(']
l2 = ['n', 'o', 'o', 'o', 's', 'd', 'n']
res2 = create_response('CDMX , MTY', 'Japón', -1)

t3 = ['¡', 'CDMX', ',', 'MTY', 'a', 'Japón', '$']
l3 = ['n', 'o', 'o', 'o', 's', 'd', 'd']
res3 = create_response('CDMX , MTY', 'Japón', -1)


@pytest.mark.parametrize("tokens,labels,expected", [
    (t1, l1, res1),
    (t2, l2, res2),
    (t3, l3, res3),
])
def test_create_trip(tokens, labels, expected):
    actual = create_trip(tokens, labels)
    assert actual == expected
