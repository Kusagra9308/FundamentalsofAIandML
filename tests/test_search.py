# tests/test_search.py
import search

def test_small_map_astar():
    grid = search.parse_map_file('maps/small.map')
    start=(0,0); goal=(4,4)
    res = search.astar(grid, start, goal)
    assert res is not None
    assert res['path'][0] == start
    assert res['path'][-1] == goal
    assert res['cost'] >= 0

def test_dynamic_schedule_load():
    sched = search.load_schedule('maps/dynamic.schedule')
    # expects some entries from sample
    assert isinstance(sched, dict)
