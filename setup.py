import logging
import subprocess
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install

# Some custom command to run during setup. Typically, these commands will
# include steps to install non-Python packages.
#
# First, note that there is no need to use the sudo command because the setup
# script runs with appropriate access.
# Second, if apt-get tool is used then the first command needs to be 'apt-get
# update' so the tool refreshes itself and initializes links to download
# repositories.  Without this initial step the other apt-get install commands
# will fail with package not found errors. Note also --assume-yes option which
# shortcuts the interactive confirmation.
#
# The output of custom commands (including failures) will be logged.

#
# class CustomCommands(install):
#   """A setuptools Command class able to run arbitrary commands."""
#
#   def RunCustomCommand(self, command_list):
#     print 'Running command: %s' % command_list
#     p = subprocess.Popen(
#         command_list,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT)
#     # Can use communicate(input='y\n'.encode()) if the command run requires
#     # some confirmation.
#     stdout_data, _ = p.communicate()
#     print 'Command output: %s' % stdout_data
#     logging.info('Log command output: %s', stdout_data)
#     if p.returncode != 0:
#       raise RuntimeError('Command %s failed: exit code: %s' %
#                          (command_list, p.returncode))
#
#   def run(self):
#     self.RunCustomCommand(['apt-get', 'update'])
#     self.RunCustomCommand(
#           ['sudo', 'apt-get', 'install', '-y', 'python-tk'])
#
#     install.run(self)


REQUIRED_PACKAGES = [
    'tensorflow==1.0.1',
    'matplotlib==1.2.0',
]

setup(
    name='trainer',
    version='0.3',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Application for images colorization',
    requires=[]
)
