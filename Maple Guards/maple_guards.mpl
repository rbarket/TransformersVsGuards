# This file contains the guard implementations for various integration methods in Maple 2024

# Trager Guard
trager_guard := proc(data)
	local expr;
    expr := data[1]: # get integrand

    # algebraic function in RootOf or radical notation
    if type(expr, 'radalgfun') then
		return 1:	
	end if:
	return 0

end proc:

# Gosper Guard
# https://fr.maplesoft.com/support/help/maple/view.aspx?path=DEtools%2fIsHyperexponential
gosper_guard := proc(data)
    local expr;
    expr := data[1]: # get integrand
	if DEtools:-IsHyperexponential(expr, x) then
		return 1:
	end if:	
	return 0:

end proc:

# Elliptic Guard
elliptic_guard := proc(data)
    local expr, partslist;
    expr := data[1]: # get integrand

    partslist := `int/ellalg/elltype`(expr, 'freeof'(x), x);
    if partslist = FAIL then
        return 0;
    end if;
    return 1;
end proc:


# PseudoElliptic
pseudoelliptic_guard := proc(data)
    local expr;
    expr := data[1]: # get integrand

    # Is it a radical function 
	if type(expr, radfun(anything,x)) then
		return 1;
	end if:
	return 0:

end proc:

# MeijerG
meijerg_guard := proc(data)
    local expr;
    expr := data[1]: # get integrand

    # Is it a radical function 
	if type(expr, `^`('polynom'('anything', x), 'fraction')) then
		return 1;
	end if:
	return 0:

end proc:
