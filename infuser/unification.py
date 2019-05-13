from typing import Iterable, Mapping

from .abstracttypes import Type, DataFrameType, SymbolicAbstractType
from .rules import TypeEqConstraint


# returns an abstract type depending on the abstract types that have previously
# been assigned to left and right
# if no abstract type has been assigned yet, assigns a new one
def get_symbolic_at(left: Type, right: Type, mapping: Mapping[Type, Type]):
    m_l = mapping.get(left)
    m_r = mapping.get(right)
    
    if m_l == None and m_r == None:
        return SymbolicAbstractType()
    elif m_l == None:
        return m_r
    elif m_r == None:
        return m_l
    else:
        return m_r
    
# substitues for the columns that have abstract type either left or right
# their old abstract type with the new abstract type at
def substitute_df_cols(df_type: DataFrameType, left: Type, right: Type,
                       at: Type, mapping: Mapping[Type, Type]):
    
    for name, a_type in df_type.column_types.items():
        if a_type == left or a_type == right:
            df_type.column_types[name] = at
            mapping[a_type] = at
    
    return df_type, mapping


def unify(matches: Iterable[TypeEqConstraint]) -> Mapping[Type, Type]:
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
        
        # two symbolic types => map to same type and map every occurence of left
        # and right to that same type too
        if isinstance(right, SymbolicAbstractType) and isinstance(left, SymbolicAbstractType):
            at = get_symbolic_at(left, right, mapping)
            mapping[left] = at
            mapping[right] = at

            for m_1 in matches:  # apply the AT to all TypeEqConstraint entries
                
                m_1_l = m_1.left
                m_1_r = m_1.right
                
                # substitution necessary?
                same_left = m_1_l.__eq__(left) or m_1_l.__eq__(right)
                same_right = m_1_r.__eq__(left) or m_1_r.__eq__(right)
                
                # if we encounter a DF we need to go in there
                if isinstance(m_1_l, DataFrameType):
                    m_1.left, mapping = substitute_df_cols(m_1_l, left, right, at, mapping)
                elif same_left:
                    mapping[m_1_r] = at
                
                if isinstance(m_1_r, DataFrameType):
                    m_1.right, mapping = substitute_df_cols(m_1_r, left, right, at, mapping)
                elif same_right:
                    mapping[m_1_l] = at
                
        

        
        # two DF-Types, make sure the columns have corresponding Abstract types
        elif isinstance(right, DataFrameType) and isinstance(left, DataFrameType):
        # TODO this is still incomplete...
        # additionally to adding the columns in the DFType
        # we should add the DFType itself too
            for name, a_type_r in right.column_types.items():
                a_type_l = left.column_types[name]
            
                n_at = get_symbolic_at(a_type_l, a_type_r, mapping)
                mapping[a_type_l] = n_at
                mapping[a_type_r] = n_at


        else:  # do we have to conisder pandas ATs too?
            raise NotImplementedError()
        
    return mapping
    
    
    
    
    
    
    
    
    
'''
    while len(match_list) > 0:
        # for the first entry in matches....
        m = match_list.pop(0)
        left = m.left
        right = m.right
        
        mapped_left = mapping.get(left)
        mapped_right = mapping.get(right)
        
        # ... make sure we have the new AT in the result
        if mapped_left == None and mapped_right == None:
            at = Type() # TODO: how to create correct new AT
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
'''
