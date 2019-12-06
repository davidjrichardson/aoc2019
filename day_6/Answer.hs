import Input

import Data.Tree
import Data.List
import Data.List.Split

getChildren :: String -> [String]
getChildren parent = (map (snd)) . (filter (\a -> (fst a) == parent)) $ pairs

-- Question 1
tree = unfoldTree (\x -> (x, getChildren x)) $ "COM"
result_1 = (sum) . (map (\a -> fst a * (length $ snd a))) $ zip [0..] $ levels $ tree

-- Question 2
-- Function taken from stack overflow; can't remember where
pathsToNode :: Eq a => a -> Tree a -> [[a]]
pathsToNode x (Node y ns) = [[x] | x == y] ++ map (y:) (pathsToNode x =<< ns)

comToSan = (pathsToNode "SAN" tree) !! 0
comToMe = (pathsToNode "YOU" tree) !! 0

lastNode = last $ intersect comToSan comToMe
subtree = unfoldTree (\x -> (x, getChildren x)) $ lastNode

pathToSan = (pathsToNode "SAN" subtree) !! 0
pathToMe = (pathsToNode "YOU" subtree) !! 0

fullPath = (pathToSan \\ ["SAN"]) `union` (pathToMe \\ ["YOU"])
result_2 = (length fullPath) -  -- Don't need the final hop since "YOU" can indirectly orbit "SAN"