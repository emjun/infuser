from typing import Iterable, Mapping

from .abstracttypes import AbstractType
from .rules import RuleMatch


def unify(matches: Iterable[RuleMatch]) -> Mapping[AbstractType, AbstractType]:
    """Create a mapping from the original to unified abstract types.

    These new abstract types should satisfy the equality constraints
    embodied by `matches`.
    """
    
    match_list = list(matches)
    mapping : Mapping = {}
    
    
    while len(match_list) > 0:
        # for the first entry in matches....
        m = match_list.pop(0)
        left = m.left
        right = m.right
        
        mapped_left = mapping.get(left)
        mapped_right = mapping.get(right)
        
        # ... make sure we have the new AT in the result
        if mapped_left == None and mapped_right == None:
            at = AbstractType() # TODO: how to create correct new AT
            mapping[left] = at
            mapping[right] = at
        elif mapped_left == None:
            mapping[left] = mapping.get(right)
        elif mapped_right == None:
            mapping[right] = mapping.get(left)
        else:
            assert(mapped_left == mapped_right)
            
        at = mapping.get(left)
        
        # now for all other matches, set the mapped AT to that of the first element
        # given we have found a shared AT from the analysis
        for o_m in matches: 
            l = o_m.left
            r = o_m.right
            
            if l == left or l == right or r == left or r == right:
                mapping[l] = at
                mapping[r] = at

    return mapping
    #raise NotImplementedError()
