# Unit tests
import guerilla.train.train_test as tt
import guerilla.play.play_test as pt
import guerilla.data_test as dt

def main():

    if dt.run_data_tests() and pt.run_play_tests() and tt.run_train_tests():
        print "All tests passed"
    else:
        print "You broke something - go fix it"


if __name__ == '__main__':
    main()
