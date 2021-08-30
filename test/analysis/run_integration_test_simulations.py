#!/usr/bin/env python3
"""
Run test simulations in IDSimF to generate current trajectory files
"""

import sys
import os.path as path
import subprocess

def main():
    print("hello")
    print(sys.argv[1])

    idsimf_build_base_path = sys.argv[1]

    # BT-QITSim executable:
    qit_sim_exe = path.join(idsimf_build_base_path, 'applications', 'ionTraps', 'BT-QITSim', 'BT-QITSim')
    sim_input = path.join('.', 'integration', 'inputs', 'qitSim_2021_08_30_001_conf.json')
    sim_result_path = path.join('.', 'integration', 'sim_results', 'qitSim_2021_08_30_001')

    subprocess.run(qit_sim_exe+' '+sim_input+' '+sim_result_path, shell=True, check=True)





if __name__ == "__main__":
    main()