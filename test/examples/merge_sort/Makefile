CC = gcc
PSORT_EXEC = psort
FLAGS = -std=c99 -D_BSD_SOURCE -lm -fopenmp

all : $(PSORT_EXEC)


$(PSORT_EXEC) : 
	$(CC) -o $(PSORT_EXEC) *.c $(FLAGS)

clean :
	rm -rf $(PSORT_EXEC)

run : clean $(PSORT_EXEC)
	./$(PSORT_EXEC)
