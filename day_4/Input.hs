module Input where
    import Data.List.Split

    rawInput = "146810-612564"

    inputNumbers :: [Int]
    inputNumbers = (map read) . (splitOn "-") $ rawInput
    lowerBound = inputNumbers !! 0
    upperBound = inputNumbers !! 1