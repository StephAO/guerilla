# Unit tests
import guerilla.train.train_test as tt
import guerilla.play.play_test as pt

def main():

    if pt.run_play_tests() and tt.run_train_tests():
        print "All tests passed"
    else:
        print "You broke something - go fix it"


if __name__ == '__main__':
    main()
