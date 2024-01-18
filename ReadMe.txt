                        NAME : NALLABOTHU SHASHANK
                        UTA ID : 1002164633

Progarmming Language Used: Python 

Code Structure: I Used a Single Python file name Final.py and inside of that I a total of 7 algorithms with funtion names 1. BFS 2.uniform_cost_serach 3.Greedy 4.a_star 5.dfs 6.ids 7.dls

Explanation of all Alog's:

We have to run te code from commad line argument there the order is <file-name> <start.txt> <goal.txt> <alog name> <true-false for dump file this how we shouldrun this code

Going into the techinal Part of the Code here lets take example of BFS all the other alogs also followed the same pattern

So here the function in BFS is bfs here we defined all the variables and the next function we have is 
print_solution_<algoname> :- and this function handles all the print statements that we are printing 
write_trace_file_<algo name> :- and this file handles everything about the file handling writing everthing in the file
generate_moves_<algoname> :- this function handles all the moves of the puzzle
get_black_pos_<algoname> - this function tells the blank posistion 
update_state_info_<algoname> :- this function updates the fringe size ad other things
this above function is for all the algos with some minor chnges 


special mention for DLS: 

while implementing DLS we need to give 3 argumnents only and then enter the the depth of the DLS that we need 

and we have different read_input_txt_to_list_format type funnctions this one is to convert our matrix frm start and goal inputs to a single line read_input_txt_to_list_format

and finally to implement this we have a main function there we check with command line arguments that are entered and we use the matrix to single line conversion fuctions and call our main algo function
















