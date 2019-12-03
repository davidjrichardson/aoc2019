-- Day one of the Advent of Code
import Input
import qualified Data.Map as Map

-- Question 1
intcode :: [Int] -> (Map.Map Int Int) -> (Map.Map Int Int)
intcode codes state
    | oper -- Pattern match over oper; what do
    | otherwise     = state
    where
        oper = take 4 codes


state :: Map.Map Int Int
state = Map.fromList []