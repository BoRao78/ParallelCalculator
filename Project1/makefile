MPICC  = mpicc
CFLAGS = # -O
FILENAME1 = Project1A2
FILENAME2 = Project1Fox
STD = c99

all:    Project1A2 Project1Fox

clean:
	rm -f *.[oadD] *.[mM][oO][dD] ${FILENAME1} ${FILENAME2}

Project1A2:	${FILENAME1}.c
		${MPICC} ${CFLAGS} -std=${STD} -o ${FILENAME1} ${FILENAME1}.c -lm

Project1Fox:	${FILENAME2}.c
		${MPICC} ${CFLAGS} -std=${STD} -o ${FILENAME2} ${FILENAME2}.c -lm
