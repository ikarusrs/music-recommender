CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
TARGET    = music-recommender
OBJS      = main.o

TESTBIN   = tests/tests

.PHONY: all clean start test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

start: $(TARGET)
	./$(TARGET)

# run from the repo root, the tests read tests/fixtures/ relative to it
test: $(TARGET) $(TESTBIN)
	$(TESTBIN)

$(TESTBIN): tests/tests.cpp core.h music.h
	$(CXX) $(CXXFLAGS) -I. -o $@ tests/tests.cpp

main.o: main.cpp core.h music.h
	$(CXX) $(CXXFLAGS) -c main.cpp

clean:
	rm -f $(TARGET) main.o $(TESTBIN)
