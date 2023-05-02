import heapq 

def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)

def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt

def lazy_greedy(F, V, B):
    """
    Args
    - F: Submodular Objective
    - V: list of indices of columns of Similarity Matrix
    - B: Budget of subset (int)
    """
    sset = []

    order = []
    heapq._heapify_max(order)
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        #if improv >= 0:
        if not order:
            sset.append(el[1])
            F.add(el[1])
        else:
            top = _heappop_max(order)
            if improv >= top[0]:
                sset.append(el[1])
                F.add(el[1])
            else:
                _heappush_max(order, (improv, el[1]))
            _heappush_max(order, top)
    return sset