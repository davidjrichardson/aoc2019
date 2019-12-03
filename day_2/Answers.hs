-- Day one of the Advent of Code
import Input
import qualified Data.Map as Map

-- Question 1
addcode :: Int -> Int -> Int -> (Map.Map Int Int) -> (Map.Map Int Int)
addcode verb noun place map = Map.insert place (map Map.! noun + map Map.! verb) map

multcode :: Int -> Int -> Int -> (Map.Map Int Int) -> (Map.Map Int Int)
multcode verb noun place map = Map.insert place (map Map.! noun * map Map.! verb) map

intcode :: [Int] -> (Map.Map Int Int) -> (Map.Map Int Int)
intcode (1  : verb : noun : place : rest) state = intcode rest (addcode verb noun place state)
intcode (2  : verb : noun : place : rest) state = intcode rest (multcode verb noun place state)
intcode codes state = state

-- Create a Map from the codes so I can refer by index
state :: Map.Map Int Int
state = Map.fromList (zip [0..1000] values)

result = (intcode values state) Map.! 0