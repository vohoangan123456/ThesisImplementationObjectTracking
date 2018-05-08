class globalConfig(dict):
    #A super duper fancy custom made CLI argument handler!!
    _descriptions = {'help, --h, -h': 'show this super helpful message and exit'}
    
    def setDefaults(self):
        self.define('video1', './videos/video1.avi', 'path to testing directory with first video')
        self.define('video2', './videos/video2.avi', 'path to testing directory with second video')
        self.define('type', False, 'the cameras setting type (false is parallel or true is not)')
        self.define('thresholdY', 20, 'threshold of different y axis to check an object behind another')
        self.define('thresholdtrack', 4, 'threshold used for check two object is different or not when tracking')
        self.define('thresholdmatch', 5, 'threshold used for match two object in the fov')

    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    #def parseArgs(self, args):
    #    i = 1
    #    while i < len(args):
    #        if args[i] == '-h' or args[i] == '--h' or args[i] == '--help':
    #            self.help() #Time for some self help! :)
    #        if len(args[i]) < 2:
    #            print('ERROR - Invalid argument: ' + args[i])
    #            print('Try running flow --help')
    #            exit()
    #        argumentName = args[i][2:]
    #        if isinstance(self.get(argumentName), bool):
    #            if not (i + 1) >= len(args) and (args[i + 1].lower() != 'false' and args[i + 1].lower() != 'true') and not args[i + 1].startswith('--'):
    #                print('ERROR - Expected boolean value (or no value) following argument: ' + args[i])
    #                print('Try running flow --help')
    #                exit()
    #            elif not (i + 1) >= len(args) and (args[i + 1].lower() == 'false' or args[i + 1].lower() == 'true'):
    #                self[argumentName] = (args[i + 1].lower() == 'true')
    #                i += 1
    #            else:
    #                self[argumentName] = True
    #        elif args[i].startswith('--') and not (i + 1) >= len(args) and not args[i + 1].startswith('--') and argumentName in self:
    #            if isinstance(self[argumentName], float):
    #                try:
    #                    args[i + 1] = float(args[i + 1])
    #                except:
    #                    print('ERROR - Expected float for argument: ' + args[i])
    #                    print('Try running flow --help')
    #                    exit()
    #            elif isinstance(self[argumentName], int):
    #                try:
    #                    args[i + 1] = int(args[i + 1])
    #                except:
    #                    print('ERROR - Expected int for argument: ' + args[i])
    #                    print('Try running flow --help')
    #                    exit()
    #            self[argumentName] = args[i + 1]
    #            i += 1
    #        else:
    #            print('ERROR - Invalid argument: ' + args[i])
    #            print('Try running flow --help')
    #            exit()
    #        i += 1
