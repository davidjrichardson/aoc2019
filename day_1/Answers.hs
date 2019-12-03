-- Day one of the Advent of Code
import Input

-- Fuel function for iterating over the input in q1 with map/foldr
fuel :: Integer -> Integer
fuel w = (w `div` 3) - 2

-- Recursive fuel function for iterating over the input in q2 with map/foldr
fuelr :: Integer -> Integer
fuelr w
    | w' <=0 = 0
    | otherwise = w' + fuelr w'
    where w' = fuel w

-- Question 1
firstAnswer = (foldr (+) 0) . (map fuel) $ values
-- Question 2
secondAnswer = (foldr (+) 0) . (map fuelr) $ values