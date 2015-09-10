"""The Game of """
import collections
from math import factorial

from dice import make_test_dice, four_sided, six_sided
from ucb import main

GOAL_SCORE = 100  # The goal of Hog is to score 100 points.

probability_dict = None
best_chances_dict = None
primes = None
vs_former_final_strategy_file = "best_chances1.txt"
vs_always_5_file = "against_5_with_primes_fix.txt"
# vs_always_5_file = "against_5_2.txt"
# probabilities_file_name = "probabilities_2.txt"
probabilities_file_name = "probabilities.txt"


# region Strategy Lib
def get_primes():
    global primes
    if primes is None:
        with open("primes1000.txt") as file:
            primes = list(map(int, file.readlines()))
    return primes


def is_prime(n):
    return n in get_primes()


def previous_prime(n):
    """

    :param n:
    :return:
    >>> previous_prime(3)
    2
    >>> previous_prime(5)
    3
    """
    i = get_primes().index(n)
    if i == 0:
        return None
    else:
        return get_primes()[i - 1]


def next_prime(x):
    """

    :param x:
    :return:
    >>> next_prime(2)
    3
    >>> next_prime(3)
    5
    >>> next_prime(5)
    7
    >>> next_prime(7)
    11
    >>> next_prime(4)
    Traceback (most recent call last):
      ...
    AssertionError
    """
    assert type(x) == int
    assert is_prime(x)
    i = get_primes().index(x)
    return get_primes()[i + 1]


def is_win(score, opponent_score, goal=100):
    return score >= goal > opponent_score


def is_loss(score, opponent_score, goal=100):
    return is_win(opponent_score, score, goal)


def free_bacon(opponent_score):
    return max(map(int, str(opponent_score))) + 1


def select_dice(score, opponent_score):
    """Select six-sided dice unless the sum of SCORE and OPPONENT_SCORE is a
    multiple of 7, in which case select four-sided dice (Hog wild).
    """
    # BEGIN Question 3
    if (score + opponent_score) % 7 == 0:
        return four_sided
    return six_sided
    # END Question 3


def is_swap(score0, score1):
    """Returns whether the last two digits of SCORE0 and SCORE1 are reversed
    versions of each other, such as 19 and 91.
    >>> is_swap(19, 91)
    True
    >>> is_swap(9, 90)
    True
    >>> is_swap(90, 9)
    True
    >>> is_swap(1, 1)
    False
    """
    # BEGIN Question 4
    a = score0 % 100
    b1 = score1 % 10
    b2 = score1 % 100 // 10
    return a == b1 * 10 + b2
    # END Question 4


def last_2(score):
    """

    :param score:
    :return:
    >>> last_2(180)
    '80'
    """
    return ('0' + str(score % 100))[-2:]


def hogtimus_prime(turn_score):
    if is_prime(turn_score):
        return next_prime(turn_score)
    return turn_score


# endregion

# region Basic Strategies
def bacon_strategy(score, opponent_score, margin=8, num_rolls=5):
    """This strategy rolls 0 dice if that gives at least MARGIN points,
    and rolls NUM_ROLLS otherwise.

    >>> bacon_strategy(20, 20, margin=5, num_rolls=4)
    0
    >>> bacon_strategy(20, 30, margin=5, num_rolls=4)
    4

    """
    # BEGIN Question 8
    if hogtimus_prime(free_bacon(opponent_score)) >= margin:
        return 0
    return num_rolls
    # END Question 8


def swap_strategy(score, opponent_score, num_rolls=5):
    """This strategy rolls 0 dice when it results in a beneficial swap and
    rolls NUM_ROLLS otherwise.
    """
    # BEGIN Question 9
    turn_score = hogtimus_prime(free_bacon(opponent_score))
    if is_swap(score + turn_score, opponent_score) and opponent_score > (score + turn_score):
        return 0
    return num_rolls
    # END Question 9


def always_roll(n):
    """Return a strategy that always rolls N dice.

    A strategy is a function that takes two total scores as arguments
    (the current player's score, and the opponent's score), and returns a
    number of dice that the current player will roll this turn.

    >>> strategy = always_roll(5)
    >>> strategy(0, 0)
    5
    >>> strategy(99, 99)
    5
    """

    def strategy(score, opponent_score):
        return n

    return strategy


# endregion

# region Simulation
def roll_dice(num_rolls, dice=six_sided):
    """Simulate rolling the DICE exactly NUM_ROLLS times. Return the sum of
    the outcomes unless any of the outcomes is 1. In that case, return 0.
    """
    # These assert statements ensure that num_rolls is a positive integer.
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls > 0, 'Must roll at least once.'
    # BEGIN Question 1
    rolls = [dice() for i in range(num_rolls)]
    if 1 in rolls:
        return 0
    return sum(rolls)
    # END Question 1


def take_turn(num_rolls, opponent_score, dice=six_sided):
    """Simulate a turn rolling NUM_ROLLS dice, which may be 0 (Free bacon).

    num_rolls:       The number of dice rolls that will be made.
    opponent_score:  The total score of the opponent.
    dice:            A function of no args that returns an integer outcome.

    >>> take_turn(0, 60)
    11
    >>> take_turn(1, 0, make_test_dice(3))
    5
    """
    assert type(num_rolls) == int, 'num_rolls must be an integer.'
    assert num_rolls >= 0, 'Cannot roll a negative number of dice.'
    assert num_rolls <= 10, 'Cannot roll more than 10 dice.'
    assert opponent_score < 100, 'The game should be over.'
    # BEGIN Question 2
    if num_rolls == 0:
        turn_score = free_bacon(opponent_score)
    else:
        turn_score = roll_dice(num_rolls, dice)
    turn_score = hogtimus_prime(turn_score)
    return turn_score
    # END Question 2


def other(who):
    """Return the other player, for a player WHO numbered 0 or 1.

    >>> other(0)
    1
    >>> other(1)
    0
    """
    return 1 - who


def play(strategy0, strategy1, score0=0, score1=0, goal=GOAL_SCORE):
    """Simulate a game and return the final scores of both players, with
    Player 0's score first, and Player 1's score second.

    A strategy is a function that takes two total scores as arguments
    (the current player's score, and the opponent's score), and returns a
    number of dice that the current player will roll this turn.

    strategy0:  The strategy function for Player 0, who plays first
    strategy1:  The strategy function for Player 1, who plays second
    score0   :  The starting score for Player 0
    score1   :  The starting score for Player 1
    """
    who = 0  # Which player is about to take a turn, 0 (first) or 1 (second)
    # BEGIN Question 5
    while score0 < goal and score1 < goal:
        if who == 0:
            score, opponent_score, strategy = score0, score1, strategy0
        else:
            score, opponent_score, strategy = score1, score0, strategy1

        num_rolls = strategy(score, opponent_score)

        turn_score = take_turn(num_rolls, opponent_score, select_dice(score, opponent_score))

        if turn_score == 0:
            opponent_score += num_rolls
        else:
            score += turn_score

        if is_swap(score, opponent_score):
            score, opponent_score = opponent_score, score

        if who == 0:
            score0, score1 = score, opponent_score
        else:
            score0, score1 = opponent_score, score
        who = other(who)
    # END Question 5
    return score0, score1


# endregion

# region Experimenting
def make_averaged(fn, num_samples=1000):
    """Return a function that returns the average_value of FN when called.

    To implement this function, you will have to use *args syntax, a new Python
    feature introduced in this project.  See the project description.

    >>> dice = make_test_dice(3, 1, 5, 6)
    >>> averaged_dice = make_averaged(dice, 1000)
    >>> averaged_dice()
    3.75
    >>> make_averaged(roll_dice, 1000)(2, dice)
    5.5

    In this last example, two different turn scenarios are averaged.
    - In the first, the player rolls a 3 then a 1, receiving a score of 0.
    - In the other, the player rolls a 5 and 6, scoring 11.
    Thus, the average value is 5.5.
    Note that the last example uses roll_dice so the hogtimus prime rule does
    not apply.
    """
    # BEGIN Question 6

    def averaged(*args):
        total = 0
        for i in range(0, num_samples):
            total += fn(*args)
        return total / num_samples

    return averaged
    # END Question 6


def max_scoring_num_rolls(dice=six_sided, num_samples=1000):
    """Return the number of dice (1 to 10) that gives the highest average turn
    score by calling roll_dice with the provided DICE over NUM_SAMPLES times.
    Assume that dice always return positive outcomes.

    >>> dice = make_test_dice(3)
    >>> max_scoring_num_rolls(dice)
    10
    """
    # BEGIN Question 7
    return \
        max({num_rolls: make_averaged(roll_dice, num_samples)(num_rolls, dice) for num_rolls in range(1, 11)}.items(),
            key=lambda x: x[1])[0]

    # END Question 7


def winner(strategy0, strategy1, score0=0, score1=0):
    """Return 0 if strategy0 wins against strategy1, and 1 otherwise."""
    score0, score1 = play(strategy0, strategy1, score0, score1)
    if score0 > score1:
        return 0
    else:
        return 1


def average_win_rate(strategy, score0=0, score1=0, num_samples=1000, baseline=always_roll(5)):
    """Return the average win rate of STRATEGY against BASELINE. Averages the
    winrate when starting the game as player 0 and as player 1.
    """
    win_rate_as_player_0 = 1 - make_averaged(winner, num_samples)(strategy, baseline, score0, score1)
    win_rate_as_player_1 = make_averaged(winner, num_samples)(baseline, strategy, score0, score1)

    return (win_rate_as_player_0 + win_rate_as_player_1) / 2


def run_experiments():
    """Run a series of strategy experiments and report results.
    >>> run_experiments()
    """
    if False:  # Change to False when done finding max_scoring_num_rolls
        six_sided_max = max_scoring_num_rolls(six_sided)
        print('Max scoring num rolls for six-sided dice:', six_sided_max)
        four_sided_max = max_scoring_num_rolls(four_sided)
        print('Max scoring num rolls for four-sided dice:', four_sided_max)

    if False:  # Change to True to test always_roll(8)
        print('always_roll(8) win rate:', average_win_rate(always_roll(8)))

    if False:  # Change to True to test bacon_strategy
        print('bacon_strategy win rate:', average_win_rate(bacon_strategy))

    if False:  # Change to True to test swap_strategy
        print('swap_strategy win rate:', average_win_rate(swap_strategy))


# endregion

# region Command Line Interface
@main
def run(*args):
    """Read in the command-line argument and calls corresponding functions.

    This function uses Python syntax/techniques not yet covered in this course.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Play Hog")
    parser.add_argument('--run_experiments', '-r', action='store_true',
                        help='Runs strategy experiments')

    args = parser.parse_args()

    if args.run_experiments:
        run_experiments()


# endregion

# region Final Strategy
def get_score_map(n, dice, opponent_score):
    """


    :param opponent_score:
    :param n:
    :return:
    >>> get_score_map(3,four_sided, 95)
    """
    if n == 0:
        return {free_bacon(opponent_score): 1}
    if dice == six_sided:
        sides = 6
    else:
        sides = 4

    score_map = collections.defaultdict(lambda: 0)
    score_map[0] = prob_rolling_1(n, sides)
    for p in range(n * 2, n * sides + 1):
        if p == 2:
            score_map[p] = 0  # 2 is the first prime, so there's no way we can score 2
        elif is_prime(p):
            next_p = next_prime(p)
            score_map[next_p] = get_probability(p, n, sides)
        else:
            score_map[p] = get_probability(p, n, sides)

    return score_map


def prob_rolling_1(n, sides):
    """

    :param n:
    :param sides:
    :return:
    >>> prob_rolling_1(1, 4)
    0.25
    >>> prob_rolling_1(2, 4)
    0.4375
    """
    return 1 - (((sides - 1) / sides) ** n)


def ncr(n, r):
    """

    :param n:
    :param r:
    :return:
    >>> ncr(12, 5)
    >>> ncr(10, 7)
    >>> ncr(8, 3)
    """
    f = factorial
    return f(n) / f(r) / f(n - r)


def probability_parse(line):
    line_split = line.split(",")
    return tuple(map(int, line_split[:3])), float(line_split[3])


def get_probability(p, n, s):
    if (p, n, s) in get_probability_dict():
        return probability_dict[(p, n, s)]
    with open(probabilities_file_name, "a") as file:
        probability = ((s - 1) / s) ** n * get_probability_normal_dice(p - n, n, s - 1)
        file.write("%d,%d,%d,%f\n" % (p, n, s, probability))
        get_probability_dict()[(p, n, s)] = probability
        return probability


def get_probability_normal_dice(p, n, s):
    """

    :param p: sum
    :param n: num rolls
    :param s: num sides
    :return:
    >>> get_probability_normal_dice(7, 2, 6)
    0.16666666666666666
    """
    return 1 / s ** n * sum([(-1) ** k * ncr(n, k) * ncr(p - s * k - 1, n - 1) for k in range(0, int((p - n) / s) + 1)])


def get_probability_dict():
    global probability_dict
    if probability_dict is None:
        with open(probabilities_file_name, "r") as file:
            lines = file.readlines()
            probability_dict = dict(probability_parse(line) for line in lines)
    return probability_dict


def get_best_chances_dict(against=always_roll(5)):
    global best_chances_dict
    if against == final_strategy:
        file_name = vs_former_final_strategy_file
    else:
        file_name = vs_always_5_file
    if best_chances_dict is None:
        with open(file_name, "r") as file:
            lines = file.readlines()
            best_chances_dict = dict(parse(line) for line in lines)
    return best_chances_dict


def get_outcome_map(n, score, opponent_score):
    """

    :param n:
    :param score:
    :param opponent_score:
    :return:
    """
    score_map = get_score_map(n, select_dice(score, opponent_score), opponent_score)
    if len(score_map) == 1:
        turn_score = next(iter(score_map.keys()))
        new_score, new_opponent_score = score, opponent_score
        if turn_score == 0:
            new_opponent_score += n
        else:
            new_score += turn_score
        if is_swap(new_score, new_opponent_score):
            new_score, new_opponent_score = new_opponent_score, new_score
        return {(new_score, new_opponent_score): 1.0}

    outcome_map = collections.defaultdict(lambda: 0)
    for turn_score in score_map:
        new_score, new_opponent_score = score, opponent_score
        if turn_score == 0:
            new_opponent_score += n
        else:
            new_score += turn_score
        if is_swap(new_score, new_opponent_score):
            new_score, new_opponent_score = new_opponent_score, new_score
        outcome_map[(new_score, new_opponent_score)] += score_map[turn_score]
    return outcome_map


def throws(f, *args):
    try:
        f(*args)
    except:
        return True
    return False


def get_best_move(score, opponent_score, against=always_roll(5)):
    """

    :param score:
    :param opponent_score:
    :return:
    >>> get_best_move(99, 99)
    (0, 1.0)
    >>> get_best_move(90, 99)
    (0, 1.0)
    """
    if (score, opponent_score) in get_best_chances_dict(against):
        return get_best_chances_dict()[(score, opponent_score)]
    best_n, best_chance = 0, 0
    for n in range(0, 11):
        outcome_chance = get_outcome_chance(score, opponent_score, n, against)

        if outcome_chance > best_chance:
            best_n, best_chance = n, outcome_chance

    if against == final_strategy:
        file_name = vs_former_final_strategy_file
    else:
        file_name = vs_always_5_file
    with open(file_name, "a") as file:
        get_best_chances_dict(against)[(score, opponent_score)] = (best_n, best_chance)
        file.write("%d,%d,%d,%f\n" % (score, opponent_score, best_n, best_chance))
    return best_n, best_chance


def get_outcome_chance(score, opponent_score, n, against):
    outcome_map = get_outcome_map(n, score, opponent_score)
    outcome_chance = 0
    for outcome in outcome_map:
        count = outcome_map[outcome]
        if is_win(outcome[0], outcome[1]):
            outcome_chance += count * 1.0
        elif is_loss(outcome[0], outcome[1]):
            outcome_chance += count * 0.0  # should just be pass
        else:
            if against == final_strategy:
                outcome_chance += count * (1 - get_best_move(outcome[1], outcome[0])[1])
            else:
                sub_outcome_chance = 0
                sub_outcome_map = get_outcome_map(5, outcome[1], outcome[0])
                for sub_outcome in sub_outcome_map:
                    sub_count = sub_outcome_map[sub_outcome]
                    if is_win(sub_outcome[1], sub_outcome[0]):
                        sub_outcome_chance += sub_count * 1.0
                    elif is_loss(sub_outcome[1], sub_outcome[0]):
                        sub_outcome_chance += sub_count * 0.0
                    else:
                        sub_outcome_chance += sub_count * get_best_move(sub_outcome[1], sub_outcome[0])[1]
                outcome_chance += count * sub_outcome_chance
    return outcome_chance


def parse(line):
    line_split = line.split(",")
    return (int(line_split[0]), int(line_split[1])), (int(line_split[2]), float(line_split[3]))


def final_strategy(score, opponent_score, against=always_roll(5)):
    """Write a brief description of your final strategy.

    """
    # BEGIN Question 10

    return get_best_move(score, opponent_score, against)[0]
    # END Question 10


def test_final_strategy(score0, score1, num_samples):
    """

    :param score0:
    :param score1:
    :param num_samples:
    :return:
    >>> test_final_strategy(0, 0, 10000)
    """
    print('(%2d, %2d):' % (score0, score1),
          average_win_rate(final_strategy, num_samples=num_samples, score0=score0, score1=score1))


def generate_strategy():
    """

    :return:
    >>> score = 99
    >>> while score >= 0:
    ...     opp = 99
    ...     while opp >= 0:
    ...         final_strategy(score, opp)
    ...         opp -= 1
    ...     score -= 1
    """

# endregion
