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

table_guards := table([
					"trager" = trager_guard,  
					"elliptic" = elliptic_guard, 
					"pseudoelliptic" = pseudoelliptic_guard, 
					"gosper" = gosper_guard]):


# read in guards
read("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Maple Guards/maple_guards.mpl"):

perfect_guards := ["gosper", "trager", "pseudoelliptic"]:

for guard in perfect_guards do
    file_path := cat("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/filtered_test/", guard, "_filtered_data.json"):
    data := Import(file_path): 

    data := [seq([parse(data[i][1]), # parse the integrand string to turn it into a Maple object
                data[i][2], # prefix notation
                data[i][3], # integral (string)
                data[i][4]], # labels of sub-algos
                i=1..numelems(data)
            )]:

    # Trager Results
    guard_results := map(table_guards[guard], data): # i.e. what did the guard predict
    real_results := map(check_algo_success, data, guard): # actual result from the guard

    printf("%s Guard Accuracy:  %.4f\n", guard, Accuracy(guard_results, real_results));
    printf("%s Guard Precision:  %.4f\n\n", guard, Precision(guard_results, real_results));
end do:





