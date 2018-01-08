import numpy as np
import matplotlib.pyplot as plt
import sys,os
from TicTacToe import TicTacToe


class TicTacToeAI:

    def __init__(self, cycles, gamesPerRound = 10, movesPerGame = 30, numberOfRandomTest = 100,  numChildren = 10, layers= 1, sigma = 0.1, filename = "defaultAI.npy",seed = 0):
        try:
            self.synaps = np.load("layers_%s_" %(layers) + filename )
            self.bias = np.load("layers_%s_" %(layers) + "bias_" + filename )
            print(self.bias)
        except:
            self.synaps = np.random.normal(0,sigma, (layers, 9,9))
            self.bias = np.random.normal(0,sigma,(layers,9))

        self.layers = layers
        self.cycles = cycles
        self.numChildren = numChildren
        self.sigma = sigma
        self.filename = filename
        self.gamesPerRound = gamesPerRound
        self.movesPerGame = movesPerGame
        self.numberOfRandomTest = numberOfRandomTest

        self.failAfterGuess = 5
        self.percentChildren = 30.0
        self.winPrice = 36
        self.losePrice = 18

        self.score = np.zeros(self.numChildren)
        self.highscore = np.zeros(cycles)

        if seed:
            np.random.seed(seed)


    def trainAI(self,savePerCycle = False,plotScore = False, randomize = True):



        self.makeChilderen(self.synaps,self.bias)

        for i in range(int(self.cycles)):
            self.score = np.zeros(self.numChildren)

            if randomize:
                for j in range(int(self.numberOfRandomTest)):
                    p1 = np.random.uniform(0,self.numChildren)
                    p2 = np.random.uniform(0,self.numChildren)

                    s = self.playGame(p1,p2)

                    self.score[int(p1)] += s[0]
                    self.score[int(p2)] += s[1]
            if not randomize:
                for p1 in xrange(self.numChildren):
                    for p2 in xrange(self.numChildren):
                        if p1!=p2:
                            s = self.playGame(p1,p2)
                            self.score[int(p1)] += s[0]
                            self.score[int(p2)] += s[1]



            indexBestChild = np.argsort(self.score)[-1]

            self.highscore[i] = self.score[indexBestChild]
            print(self.score[indexBestChild])

            if savePerCycle:
                np.save("layers_%s_" %(self.layers) + self.filename ,self.synaps[indexBestChild,:,:,:])
                np.save("layers_%s_" %(self.layers) + "bias_" + self.filename ,self.bias[indexBestChild,:,:])
            if not(i == self.cycles - 1):
                self.makeChilderen(self.synaps[indexBestChild,:,:,:],self.bias[indexBestChild,:,:])

            print("Done with Cycle %s" %i)
            print("############################")

        # np.save(self.filename,self.synaps[indexBestChild,:,:,:])
        # np.save("bias_" + self.filename,self.bias[indexBestChild,:,:])
        np.save("layers_%s_" %(self.layers) + self.filename ,self.synaps[indexBestChild,:,:,:])
        np.save("layers_%s_" %(self.layers) + "bias_" + self.filename ,self.bias[indexBestChild,:,:])

        if plotScore:
            self.plotHighScore()





    def nonlin(self,x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))


    def makeChilderen(self,bestChild,bestBias):

        newSynaps = np.zeros((self.numChildren, self.layers,9,9))
        newBias = np.zeros((self.numChildren, self.layers,9))

        for i  in range(int(np.floor(self.numChildren/self.percentChildren))):
            newSynaps[i] = bestChild
            newBias[i] = bestBias

        for i in range(int(np.ceil(self.numChildren/self.percentChildren)),int(self.numChildren)):
            newSynaps[i] = bestChild + np.random.normal(0,self.sigma, (self.layers,9,9))
            newBias[i] = bestBias + np.random.normal(0,self.sigma, (self.layers,9))



        self.synaps = newSynaps
        self.bias = newBias



    def playGame(self,p1,p2):

        finished = False
        score = [0,0]

        numRounds = 0

        game = TicTacToe()

        for i in range(int(self.gamesPerRound)):
            for j in range(int(self.movesPerGame)):
                if finished:
                    break

                numRounds +=1

                #First Player Playes
                #print self.bias[p1,:,:]
                #print self.getMove(self.synaps[p1,:,:,:], self.bias[p1,:,:], game.getBoard())
                moves = np.argsort(self.getMove(p1,self.synaps[int(p1),:,:,:], self.bias[int(p1),:,:], game.getBoard()))[::-1]


                for move in moves[0:self.failAfterGuess]:
                    if game.isMovesIsLegal(move):

                        game.makeMove(move)
                        break
                else:
                    score[1] +=self.winPrice+ numRounds
                    score[0] -= self.losePrice
                    break


                if game.checkWinner():
                    if int(game.checkWinner()) == 1:
                        score[0] += self.winPrice + numRounds#self.failAfterGuess
                        score[1] -= self.losePrice - numRounds#self.failAfterGuess
                        break
                    elif int(game.checkWinner()) == 2:
                        score[1] += self.winPrice + numRounds#self.failAfterGuess
                        score[0] -= self.losePrice - numRounds#self.failAfterGuess
                        break


                #Second Player Playes
                moves = np.argsort(self.getMove(p2,self.synaps[int(p2),:,:,:], self.bias[int(p2),:,:], game.getBoard()))[::-1]

                for move in moves:
                    if game.isMovesIsLegal(move):
                        game.makeMove(move)
                        break
                else:
                    score[1] -= self.losePrice
                    score[0] += self.winPrice+ numRounds
                    break

                if game.checkWinner():
                    if int(game.checkWinner()) == 1:
                        score[0] += self.winPrice + numRounds#self.failAfterGuess
                        score[1] -= self.losePrice- numRounds#self.failAfterGuess
                        break
                    elif int(game.checkWinner()) == 2:
                        score[1] += self.winPrice + numRounds#self.failAfterGuess
                        score[0] -= self.losePrice - numRounds #self.failAfterGuess
                        break

            if finished:
                break

        return score


    def getMove(self,player,p, pBias, board):
        board = np.where(board == (int(player) + 1)%2, -1, board)
        l = self.nonlin(np.dot(board,p[0,:,:]) + pBias[0,:])

        for i in range(1,int(self.layers)):
            l = self.nonlin(np.dot(l,p[i,:,:]) + pBias[i,:])

        return l



    def playWithAI(self):
        game = TicTacToe()

        while True:

            move = raw_input("Make move")
            if move == "q":
                break
            game.makeMove(move)


            moves = np.argsort(self.getMove(2,self.synaps, self.bias, game.getBoard()))[::-1]

            for move in moves:
                if game.isMovesIsLegal(move):
                    game.makeMove(move)
                    break
            else:
                print("The AI could not find a valide move! Train it more!")

            game.drawBoard()

            winner = game.checkWinner()
            if winner:
                print("Player %s won" %(int(winner)))
                break

    def playAiAgainstAi(self):
        game = TicTacToe()

        while True:

            moves = np.argsort(self.getMove(self.synaps, self.bias, game.getBoard()))[::-1]

            for move in moves:
                if game.isMovesIsLegal(move):
                    game.makeMove(move)
                    break
            else:
                print("The AI could not find a valide move! Train it more!")


            winner = game.checkWinner()
            if winner:
                print("Player %s won" %(int(winner)))
                break


            moves = np.argsort(self.getMove(self.synaps, self.bias, game.getBoard()))[::-1]

            for move in moves:
                if game.isMovesIsLegal(move):
                    game.makeMove(move)
                    break
            else:
                print("The AI could not find a valide move! Train it more!")

            game.drawBoard()

            winner = game.checkWinner()
            if winner:
                print("Player %s won" %(int(winner)))
                break


    def plotHighScore(self):
        plt.plot(self.highscore)
        plt.show()

        meanScore = []
        for i in range(100,len(self.highscore)-100):
            meanScore.append(np.mean(self.highscore[i-100:i+100]))
        plt.plot(meanScore)
        plt.show()



if __name__ == "__main__":
    ai = TicTacToeAI(500,sigma = 1, numChildren = 7, layers = 5)

    #print (ai.getMove(ai.synaps, ai.bias,[0,0,0,2,0,0,0,0,0]))
    #print np.argsort(ai.getMove(ai.synaps, ai.bias,[0,0,0,2,0,0,0,0,0]))[::-1]
    #ai.trainAI(plotScore = True,randomize = False)
    try:
        if sys.argv[1] == "play":
            #ai.playWithAI()
            ai.playWithAI()
        elif sys.argv[1] == "train":
            ai.trainAI(plotScore = True)
    except:
        print("No choice was made.")
