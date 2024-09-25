# Method for accuracy
Accuracy := proc(predicted, actual)
    local correct_predictions, total_predictions, i;
    correct_predictions := 0;
    total_predictions := numelems(predicted);

    for i from 1 to total_predictions do
        if predicted[i] = actual[i] then
            correct_predictions := correct_predictions + 1;
        end if;
    end do;

    return evalf(correct_predictions / total_predictions);
end proc:

# Function to calculate precision
Precision := proc(predicted, actual)
    local true_positives, false_positives, i;
    true_positives := 0;
    false_positives := 0;

    for i from 1 to numelems(predicted) do
        if predicted[i] = 1 then
            if actual[i] = 1 then
                true_positives := true_positives + 1;
            else
                false_positives := false_positives + 1;
            end if;
        end if;
    end do;

    if true_positives + false_positives = 0 then
        return 0; # Avoid division by zero
    else
        return evalf(true_positives / (true_positives + false_positives));
    end if;
end proc:

# Check Success of an algorithm
check_algo_success := proc(data, algo)
	local algo_index, targets;
	
	algo_index := table([
					"default" = 1, 
					"derivativedivides" = 2, 
					"parts" = 3, 
					"risch" = 4,
					"norman" = 5,
					"trager" = 6, 
					"parallelrisch" = 7, 
					"meijerg" = 8, 
					"elliptic" = 9, 
					"pseudoelliptic" = 10, 
					"lookup" = 11, 
					"gosper" = 12]);

	targets := data[4]; # 4th spot in list holds all the targets
	if targets[algo_index[algo]] > 0 then
		return 1:
	end if:
	return 0
	
end proc:

# Get rid of prefix for integral, not needed
removeFourth := proc(lst)
    return [op(1..3, lst), op(5..nops(lst), lst)];
end proc:


# read in guards
read("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Maple Guards/maple_guards.mpl"):

BWD := Import("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/BWD_test.json"):
FWD := Import("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/FWD_test.json"):
IBP := Import("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/IBP_test.json"):
# Make data match format of SUB, RISCH
BWD := map(removeFourth, BWD):
FWD := map(removeFourth, FWD):
IBP := map(removeFourth, IBP):

SUB := Import("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/SUB_test.json"):
RISCH := Import("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/RISCH_test.json"):

data := [op(BWD), op(FWD), op(IBP), op(SUB), op(RISCH)]:
# data := [op(SUB)]:

data := [seq([parse(data[i][1]), # parse the integrand string to turn it into a Maple object
             data[i][2], # prefix notation
             data[i][3], # integral (string)
             data[i][4]], # labels of sub-algos
             i=1..numelems(data)
           )]:

# MeijerG Results
meijerg_guard_results := map(meijerg_guard, data): # i.e. what did the guard predict
meijerg_real_results := map(check_algo_success, data, "meijerg"): # actual result from the guard

printf("MeijerG Guard Accuracy:  %.4f\n", Accuracy(meijerg_guard_results, meijerg_real_results));
printf("MeijerG Guard Precision:  %.4f\n\n", Precision(meijerg_guard_results, meijerg_real_results));

# Trager Results
trager_guard_results := map(trager_guard, data): # i.e. what did the guard predict
trager_real_results := map(check_algo_success, data, "trager"): # actual result from the guard

printf("Trager Guard Accuracy:  %.4f\n", Accuracy(trager_guard_results, trager_real_results));
printf("Trager Guard Precision:  %.4f\n\n", Precision(trager_guard_results, trager_real_results));


# Gosper Results
gosper_guard_results := map(gosper_guard, data): # i.e. what did the guard predict
gosper_real_results := map(check_algo_success, data, "gosper"): # actual result from the guard

printf("Gosper Guard Accuracy:  %.4f\n", Accuracy(gosper_guard_results, gosper_real_results));
printf("Gosper Guard Precision:  %.4f\n\n", Precision(gosper_guard_results, gosper_real_results));

# Elliptic Results
elliptic_guard_results := map(elliptic_guard, data): # i.e. what did the guard predict
elliptic_real_results := map(check_algo_success, data, "elliptic"): # actual result from the guard

printf("Elliptic Guard Accuracy:  %.4f\n", Accuracy(elliptic_guard_results, elliptic_real_results));
printf("Elliptic Guard Precision:  %.4f\n\n", Precision(elliptic_guard_results, elliptic_real_results));

# PseudoElliptic
pseudoelliptic_guard_results := map(pseudoelliptic_guard, data): # i.e. what did the guard predict
pseudoelliptic_real_results := map(check_algo_success, data, "pseudoelliptic"): # actual result from the guard

printf("Pseudoelliptic Guard Accuracy:  %.4f\n", Accuracy(pseudoelliptic_guard_results, pseudoelliptic_real_results));
printf("Pseudoelliptic Guard Precision:  %.4f\n", Precision(pseudoelliptic_guard_results, pseudoelliptic_real_results));




