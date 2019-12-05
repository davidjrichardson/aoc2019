import Input
import qualified Data.List as List


ascNumbers :: Int -> Char -> [String]
ascNumbers len n
    | len == 1  = map (\x -> [x]) $ [n..'9']
    | otherwise = foldr (++) [] [map ((:) x) $ ascNumbers (len - 1) x | x <- [n..'9']]


filterPred :: ([Int] -> Bool) -> String -> Bool
filterPred cmp num = (cmp) . (map length) . (List.groupBy (==)) $ num

allNumbers = (map show) . (takeWhile (\a -> a <= upperBound)) . (dropWhile (\a -> a < lowerBound)) . (map read) $ ascNumbers 6 '1'


-- Question 1
result_1 = length $ filter (filterPred $ any (>1)) $ allNumbers

-- Question 2
result_2 = length $ filter (filterPred $ (2 `elem`)) $ allNumbers