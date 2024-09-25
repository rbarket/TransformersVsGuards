# This file is used to filter the test data to only keep samples that were rated positive (i.e. throw out True Negatives according to guards)

# Auxillary Function to convert Maple object to string while preserving structure of list
ConvertToString := proc(listOfLists)
    local i, result;
    result := [];
    for i from 1 to numelems(listOfLists) do
        result := [op(result), [convert(listOfLists[i][1], string), op(2..numelems(listOfLists[i]), listOfLists[i])]];
    end do;
    return result;
end proc:

# Auxillary function to match shape of BWD,FWD,IBP to other data
removeFourth := proc(lst)
    return [op(1..3, lst), op(5..nops(lst), lst)];
end proc:

# Given algo and a dataset, filter the data with the guard for that algo (assuming the guard exists)
filter_with_guard := proc(data, algo)
    local table_guards, algo_guard_results, algo_real_results, algo_pos, i;

    table_guards := table([
					"trager" = trager_guard,  
					"elliptic" = elliptic_guard, 
					"pseudoelliptic" = pseudoelliptic_guard, 
					"gosper" = gosper_guard]);

    # filtered data for trager
    algo_guard_results := map(table_guards[algo], data): # i.e. what did the guard predict
    algo_real_results := map(check_algo_success, data, convert(algo,string)): # actual result from the guard


    algo_pos := [seq([op(data[i]), algo_guard_results[i]], i=1..numelems(data))]:

    # Select only the sublists where algo succeeds
    algo_pos := select(x -> x[5] = 1, algo_pos):

    algo_pos := [seq([convert(algo_pos[i][1], string), # parse the integrand string to turn it into a Maple object
                    algo_pos[i][2], # prefix notation
                    algo_pos[i][3], # integral (string)
                    algo_pos[i][4]], # labels of sub-algos
                    i=1..numelems(algo_pos)
                )]:    

    return algo_pos:

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

print("Read data"):

pseudoelliptic_filtered_data := filter_with_guard(data, "pseudoelliptic"):
print("pseudo", numelems(pseudoelliptic_filtered_data)):
Export("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/filtered_test/pseudoelliptic_filtered_data.json", pseudoelliptic_filtered_data):

trager_filtered_data := filter_with_guard(data, "trager"):
Export("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/filtered_test/trager_filtered_data.json", trager_filtered_data):

gosper_filtered_data := filter_with_guard(data, "gosper"):
Export("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/filtered_test/gosper_filtered_data.json", gosper_filtered_data):

elliptic_filtered_data := filter_with_guard(data, "elliptic"):
Export("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/filtered_test/elliptic_filtered_data.json", elliptic_filtered_data):

meijerg_filtered_data := filter_with_guard(data, "meijerg"):
Export("C:/Users/rbarket/OneDrive - Coventry University/Year 3/NeurIPS2024/Datasets/Test/filtered_test/pseudoelliptic_filtered_data.json", meijerg_filtered_data):

print("Done"):