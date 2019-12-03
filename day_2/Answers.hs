-- Day one of the Advent of Code
import Input
import qualified Data.Map as Map
import qualified Data.List as List
import qualified Data.Maybe as Maybe

-- Functions for processing Intcode
opcode :: Int -> Int -> Int -> (Map.Map Int Int) -> (Int -> Int -> Int) -> (Map.Map Int Int)
opcode verb noun place map f = Map.insert place (f (map Map.! noun) (map Map.! verb)) map

intcode :: [Int] -> (Map.Map Int Int) -> (Map.Map Int Int)
intcode (1  : verb : noun : place : rest) state = intcode rest (opcode verb noun place state (+))
intcode (2  : verb : noun : place : rest) state = intcode rest (opcode verb noun place state (*))
intcode codes state = state

-- Question 1
-- Create a Map from the codes so I can refer by index
state :: Map.Map Int Int
state = Map.fromList (zip [0..] values)

result = (intcode values state) Map.! 0

-- Question 2
cartProd xs ys = [(x,y) | x <- xs, y <- ys]

paramSpace :: [(Int, Int)]
paramSpace = cartProd [0..99] [0..99]

changeParam :: (Int, Int) -> [Int]
changeParam pair = 1:(fst pair):(snd pair):(drop 3 values)

changeState :: (Int, Int) -> (Map.Map Int Int)
changeState pair = Map.insert 2 (snd pair) (Map.insert 1 (fst pair) (Map.fromList (zip [0..] values)))

codeSpace = map changeParam paramSpace
stateSpace = map changeState paramSpace

programs = zip codeSpace stateSpace

-- Map function across (Int, Int)
testProgram program = ((intcode (fst program) (snd program)) Map.! 0)
result_2 = paramSpace !! Maybe.fromMaybe 0 (List.elemIndex values2 (map testProgram programs))