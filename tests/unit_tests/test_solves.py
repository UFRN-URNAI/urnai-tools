import pathlib
import sys

# from urnai.solves import solve_cartpole_v0, solve_cartpole_v1, solve_frozenlake, \
#     solve_simple64_veryeasy
# from urnai.solves.experiments import solve_breakout_ram_v0, solve_breakout_v0, \
#     solve_cartpole_v1_dql, solve_simple64, solve_simple64_mo

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))


def test_cartpole_v0_declaration():
    # trainer = solve_cartpole_v0.declare_trainer()
    pass


def test_cartpole_v1_declaration():
    # trainer = solve_cartpole_v1.declare_trainer()
    pass


def test_simple64_veryeasy_declaration():
    # trainer = solve_simple64_veryeasy.declare_trainer()
    pass


def test_frozenlake_declaration():
    # trainer = solve_frozenlake.declare_trainer()
    pass


def test_experiments_breakout_ram_v0_declaration():
    # trainer = solve_breakout_ram_v0.declare_trainer()
    pass


def test_experiments_breakout_v0_declaration():
    # trainer = solve_breakout_v0.declare_trainer()
    pass


def test_experiments_cartpole_v1_declaration():
    # trainer = solve_cartpole_v1_dql.declare_trainer()
    pass


def test_experiments_simple64_declaration():
    # trainer = solve_simple64.declare_trainer()
    pass


def test_experiments_simple64_mo_declaration():
    # trainer = solve_simple64_mo.declare_trainer()
    pass
