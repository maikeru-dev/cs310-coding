(define (problem taxi2)
  (:domain taxi_part2)
  
  (:objects
    taxi_1 taxi_2 - taxi
    livingstone_tower royal_college graham_hills tic barony_hall - location
    scott rajesh lingjie alice bob charlie - person
  )
  (:init
    (outsidetaxi scott)
    (plocation scott livingstone_tower)

    (outsidetaxi rajesh)
    (plocation rajesh graham_hills)

    (outsidetaxi lingjie)
    (plocation lingjie barony_hall)

    (outsidetaxi alice)
    (plocation alice royal_college)

    (outsidetaxi bob)
    (plocation bob tic)

    (outsidetaxi charlie)
    (plocation charlie graham_hills)

    (tlocation taxi_1 tic)
    (empty taxi_1)
    (tlocation taxi_2 livingstone_tower)
    (empty taxi_2)

    (connects livingstone_tower royal_college)
    (connects royal_college graham_hills)
    (connects graham_hills tic)
    (connects tic barony_hall)
    (connects barony_hall livingstone_tower)

    (connects royal_college livingstone_tower)
    (connects graham_hills royal_college)
    (connects tic graham_hills)
    (connects barony_hall tic)
    (connects livingstone_tower barony_hall)
  )
  (:goal
    (and
      (outsidetaxi scott)
      (outsidetaxi rajesh)
      (outsidetaxi lingjie)
      (outsidetaxi alice)
      (outsidetaxi bob)
      (outsidetaxi charlie)
      (plocation scott graham_hills)
      (plocation rajesh barony_hall)
      (plocation lingjie livingstone_tower)
      (plocation alice tic)
      (plocation bob royal_college)
      (plocation charlie barony_hall)
    )
  )
)